from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import GPSTAGS
from datetime import datetime
import pytz
import io
import httpx
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print(
            "Warning: GOOGLE_API_KEY environment variable not set. Number plate extraction will fail."
        )
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error configuring Google Gemini API: {e}")

EXIF_DATETIME_ORIGINAL_TAG = 36867
GPS_INFO_TAG_ID = 34853
VIENNA_TZ = pytz.timezone("Europe/Vienna")

def get_decimal_from_dms(dms, ref):
    """
    Converts GPS coordinates from DMS (Degrees, Minutes, Seconds) format to decimal degrees.
    dms: tuple of 3 rationals (degrees, minutes, seconds)
         Each rational is an object with .numerator and .denominator attributes (e.g., IFDRational).
    ref: 'N', 'S', 'E', or 'W'
    """
    # Each dms[i] is an IFDRational object (or similar) with numerator and denominator attributes
    degrees = dms[0].numerator / dms[0].denominator
    minutes = (dms[1].numerator / dms[1].denominator) / 60.0
    seconds = (dms[2].numerator / dms[2].denominator) / 3600.0

    decimal = degrees + minutes + seconds
    if ref in ["S", "W"]:
        decimal = -decimal
    return decimal


@app.post("/extract_exif_datetime/")
async def extract_exif_datetime(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    if not file.filename.lower().endswith((".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="File is not a JPG image")

    try:
        contents = await file.read()
        img_stream = io.BytesIO(contents)
        img = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Cannot identify image file. It might be corrupted or not a valid JPG.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not open image: {str(e)}")
    finally:
        await file.close()

    exif_data = img.getexif()  # Use the newer getexif() method

    if not exif_data:
        raise HTTPException(status_code=404, detail="No EXIF data found")

    # DateTimeOriginal (tag 36867) is usually in the ExifIFD.
    # The ExifIFD is pointed to by the ExifOffset tag (34665 or 0x8769) from IFD0.
    EXIF_IFD_POINTER_TAG = 34665  # Tag ID for ExifOffset, which points to the ExifIFD
    exif_ifd = exif_data.get_ifd(EXIF_IFD_POINTER_TAG)

    datetime_original_str = None
    if exif_ifd:  # Check if the ExifIFD was found
        datetime_original_str = exif_ifd.get(EXIF_DATETIME_ORIGINAL_TAG)

    if not datetime_original_str:
        raise HTTPException(
            status_code=404, detail="EXIF DateTimeOriginal tag not found"
        )

    try:
        # EXIF datetime format is 'YYYY:MM:DD HH:MM:SS'
        naive_datetime = datetime.strptime(datetime_original_str, "%Y:%m:%d %H:%M:%S")

        # Assume the naive datetime is local time. Localize it to Vienna.
        # If the EXIF time was UTC, you would first localize to UTC then convert.
        # For this requirement, we interpret it as local and directly localize to Vienna.
        # This might not be accurate if the photo was taken in a different timezone.
        # A more robust solution would require knowing the original timezone or assuming UTC.
        # However, per request, localizing directly to Vienna.
        vienna_datetime = VIENNA_TZ.localize(naive_datetime)

        # Initialize GPS related variables
        address_str = None
        lat_decimal = None
        lon_decimal = None

        raw_gps_info = exif_data.get_ifd(GPS_INFO_TAG_ID)

        if raw_gps_info:
            decoded_gps_info = {}
            for tag_id, value in raw_gps_info.items():
                tag_name = GPSTAGS.get(tag_id, tag_id)
                decoded_gps_info[tag_name] = value

            gps_latitude = decoded_gps_info.get("GPSLatitude")
            gps_latitude_ref = decoded_gps_info.get("GPSLatitudeRef")
            gps_longitude = decoded_gps_info.get("GPSLongitude")
            gps_longitude_ref = decoded_gps_info.get("GPSLongitudeRef")

            if (
                gps_latitude
                and gps_latitude_ref
                and gps_longitude
                and gps_longitude_ref
            ):
                try:
                    lat_decimal = get_decimal_from_dms(gps_latitude, gps_latitude_ref)
                    lon_decimal = get_decimal_from_dms(gps_longitude, gps_longitude_ref)
                except (TypeError, ZeroDivisionError, IndexError) as e:
                    print(f"Error converting DMS to decimal: {e}")
                    lat_decimal = None
                    lon_decimal = None

                # Reverse geocode using Nominatim if coordinates are valid
                if lat_decimal is not None and lon_decimal is not None:
                    nominatim_url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat_decimal}&lon={lon_decimal}&accept-language=de"

                    # TODO: Customize User-Agent for your application as per Nominatim's Usage Policy
                    # See: https://operations.osmfoundation.org/policies/nominatim/
                    headers = {
                        "User-Agent": "FalschparkerApp/0.1 (contact@example.com)"
                    }

                    try:
                        async with httpx.AsyncClient() as client:
                            # Timeout set to 10 seconds
                            api_response = await client.get(
                                nominatim_url, headers=headers, timeout=10.0
                            )
                        api_response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

                        address_data = api_response.json()
                        address_components = address_data.get("address", {})

                        parts = []
                        street_name = address_components.get("road")
                        if street_name:
                            house_num = address_components.get("house_number")
                            if house_num:
                                street_name += f" {house_num}"
                            parts.append(street_name)

                        postcode = address_components.get("postcode")
                        if postcode:
                            parts.append(postcode)

                        # Try to get city, then town, then village, etc.
                        city_name = (
                            address_components.get("city")
                            or address_components.get("town")
                            or address_components.get("village")
                            or address_components.get("hamlet")
                        )
                        if city_name:
                            parts.append(city_name)

                        if parts:
                            address_str = ", ".join(parts)

                    except httpx.RequestError as exc:
                        # Log this error (e.g., network issue, DNS failure)
                        print(f"Nominatim request error for {exc.request.url!r}: {exc}")
                    except httpx.HTTPStatusError as exc:
                        # Log this error (e.g., 403 Forbidden, 429 Too Many Requests, 500 Internal Server Error)
                        print(
                            f"Nominatim API error {exc.response.status_code} for {exc.request.url!r}."
                        )
                    except Exception as e:
                        # Log any other unexpected error during geocoding
                        print(f"Unexpected error during geocoding: {str(e)}")

        # Number Plate Extraction with Gemini
        number_plate_str = None
        if GOOGLE_API_KEY:
            try:
                # Ensure the image stream is reset if it was read before for Pillow
                img_stream.seek(0)
                image_bytes = img_stream.read()

                response = generate(image_bytes, GOOGLE_API_KEY)

                if response.text:
                    # Assuming the first part contains the text response
                    extracted_text = response.text.strip()
                    if extracted_text and extracted_text.upper() != "N/A":
                        number_plate_str = extracted_text
                    elif extracted_text.upper() == "N/A":
                        number_plate_str = (
                            None  # Explicitly set to None if Gemini says N/A
                        )
                    else:  # Empty response or unexpected content
                        number_plate_str = None
                        print(
                            "Gemini response for number plate was empty or unexpected."
                        )
                else:  # No parts in response
                    number_plate_str = None
                    print(
                        "Gemini response contained no parts for number plate extraction."
                    )

            except Exception as e:
                print(f"Error during number plate extraction with Gemini: {str(e)}")
                number_plate_str = None  # Ensure it's None on error
        else:
            print("Skipping number plate extraction as GOOGLE_API_KEY is not set.")

        # Prepare response data
        response_payload = {
            "filename": file.filename,
            "datetime_original_vienna": vienna_datetime.isoformat(),
        }
        if lat_decimal is not None and lon_decimal is not None:
            response_payload["latitude"] = lat_decimal
            response_payload["longitude"] = lon_decimal
        if address_str:  # Only add address if it was successfully retrieved
            response_payload["address"] = address_str
        if number_plate_str:  # Only add number plate if it was successfully extracted
            response_payload["number_plate"] = number_plate_str

        return response_payload

    except ValueError:
        raise HTTPException(
            status_code=500, detail="Invalid datetime format in EXIF data"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


def generate(image_data, api_key):
    client = genai.Client(
        api_key=api_key,
    )

    model = "gemini-2.5-flash-preview-05-20"
    prompt = (
        "Analyze this image and extract the vehicle number plate. "
        "The image contains a car. Focus on identifying the number plate text. "
        "If a number plate is clearly visible and legible, return only the characters of the number plate. "
        "If no number plate is visible, or if it's unreadable, return 'N/A'."
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=image_data,
                ),
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0,
        ),
        response_mime_type="text/plain",
    )

    return client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
