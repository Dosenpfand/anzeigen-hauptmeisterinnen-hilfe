from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import GPSTAGS
from datetime import datetime
import pytz
import io
import httpx
import os
import logging
from typing import Tuple, Any
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Rate Limiting
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_HOURS = int(os.getenv("RATE_LIMIT_HOURS", "10"))
rate_limit_str = f"{RATE_LIMIT_REQUESTS}/{RATE_LIMIT_HOURS}hour"

limiter = Limiter(key_func=get_remote_address, default_limits=[rate_limit_str])
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.mount("/static", StaticFiles(directory="static"), name="static")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning(
        "GOOGLE_API_KEY environment variable not set. Number plate extraction will fail."
    )

EXIF_DATETIME_ORIGINAL_TAG = 36867
GPS_INFO_TAG_ID = 34853
VIENNA_TZ = pytz.timezone("Europe/Vienna")
NOMINATIM_USER_AGENT = "FalschparkerApp/0.1 (falschparker@sad.bz)"
GEMINI_NO_PLATE_RESPONSE = "N/A"
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
GEMINI_PROMPT = (
    "Analyze this image and extract the vehicle number plate. "
    "The image contains a car. Focus on identifying the number plate text. "
    "If a number plate is clearly visible and legible, return only the characters of the number plate. "
    "Do not insert any special characters, symbols, emojis, white spaces, etc."
    "If no number plate is visible, or if it's unreadable, return 'N/A'."
)


def get_decimal_from_dms(dms: Tuple[Any, Any, Any], ref: str) -> float:
    """
    Converts GPS coordinates from DMS (Degrees, Minutes, Seconds) format to decimal degrees.
    dms: tuple of 3 rational-like objects (degrees, minutes, seconds)
         Each object must have .numerator and .denominator attributes.
    ref: 'N', 'S', 'E', or 'W' indicating direction.
    """
    # Each dms[i] is an object with numerator and denominator attributes
    degrees = dms[0].numerator / dms[0].denominator
    minutes = (dms[1].numerator / dms[1].denominator) / 60.0
    seconds = (dms[2].numerator / dms[2].denominator) / 3600.0

    decimal = degrees + minutes + seconds
    if ref in ["S", "W"]:
        decimal = -decimal
    return decimal


@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("static/index.html") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="Frontend not found.", status_code=404)


@app.post("/extract_info/")
@limiter.limit(rate_limit_str)
async def extract_info(request: Request, file: UploadFile = File(...)):
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

    try:  # New try block for the rest of the logic
        # Initialize all potentially EXIF-derived data to None
        vienna_datetime_iso = None
        address_str = None
        lat_decimal = None
        lon_decimal = None

        exif_data = img.getexif()  # Use the newer getexif() method

        if exif_data:
            logger.info("EXIF data found. Processing...")
            # DateTimeOriginal (tag 36867) is usually in the ExifIFD.
            # The ExifIFD is pointed to by the ExifOffset tag (34665 or 0x8769) from IFD0.
            EXIF_IFD_POINTER_TAG = (
                34665  # Tag ID for ExifOffset, which points to the ExifIFD
            )
            exif_ifd = exif_data.get_ifd(EXIF_IFD_POINTER_TAG)

            datetime_original_str = None
            if exif_ifd:  # Check if the ExifIFD was found
                datetime_original_str = exif_ifd.get(EXIF_DATETIME_ORIGINAL_TAG)

            if datetime_original_str:
                try:
                    # EXIF datetime format is 'YYYY:MM:DD HH:MM:SS'
                    naive_datetime = datetime.strptime(
                        datetime_original_str, "%Y:%m:%d %H:%M:%S"
                    )
                    # Assume the naive datetime is local time. Localize it to Vienna.
                    vienna_datetime = VIENNA_TZ.localize(naive_datetime)
                    vienna_datetime_iso = vienna_datetime.isoformat()
                except ValueError:
                    logger.warning(
                        f"Invalid datetime format in EXIF data: '{datetime_original_str}'. Skipping datetime processing."
                    )
            else:
                logger.warning(
                    "EXIF DateTimeOriginal tag not found. Skipping datetime processing."
                )

            # GPS Info Processing
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
                        lat_decimal = get_decimal_from_dms(
                            gps_latitude, gps_latitude_ref
                        )
                        lon_decimal = get_decimal_from_dms(
                            gps_longitude, gps_longitude_ref
                        )
                    except (TypeError, ZeroDivisionError, IndexError) as e:
                        logger.error(f"Error converting DMS to decimal: {e}")
                        lat_decimal = None
                        lon_decimal = None

                    # Reverse geocode using Nominatim if coordinates are valid
                    if lat_decimal is not None and lon_decimal is not None:
                        nominatim_url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat_decimal}&lon={lon_decimal}&accept-language=de"
                        headers = {"User-Agent": NOMINATIM_USER_AGENT}

                        try:
                            async with httpx.AsyncClient() as client:
                                # Timeout set to 10 seconds
                                api_response = await client.get(
                                    nominatim_url, headers=headers, timeout=10.0
                                )
                            api_response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

                            address_data = api_response.json()
                            address_components = address_data.get("address", {})

                            address_parts = []
                            street = address_components.get("road")
                            if street:
                                house_number = address_components.get("house_number")
                                full_street = (
                                    f"{street} {house_number}"
                                    if house_number
                                    else street
                                )
                                address_parts.append(full_street)

                            postcode = address_components.get("postcode")
                            if postcode:
                                address_parts.append(postcode)

                            city_keys = ["city", "town", "village", "hamlet"]
                            city_name = next(
                                (
                                    address_components.get(key)
                                    for key in city_keys
                                    if address_components.get(key)
                                ),
                                None,
                            )
                            if city_name:
                                address_parts.append(city_name)

                            if address_parts:
                                address_str = ", ".join(address_parts)
                            # else address_str remains None, which is the default

                        except httpx.RequestError as exc:
                            # Log this error (e.g., network issue, DNS failure)
                            logger.error(
                                f"Nominatim request error for {exc.request.url!r}: {exc}"
                            )
                        except httpx.HTTPStatusError as exc:
                            # Log this error (e.g., 403 Forbidden, 429 Too Many Requests, 500 Internal Server Error)
                            logger.error(
                                f"Nominatim API error {exc.response.status_code} for {exc.request.url!r}."
                            )
                        except Exception as e:
                            # Log any other unexpected error during geocoding
                            logger.error(f"Unexpected error during geocoding: {str(e)}")
            else:  # This else corresponds to `if raw_gps_info:`
                logger.info(
                    "GPS metadata (tag GPS_INFO_TAG_ID) not found in EXIF. Skipping GPS processing."
                )
                # lat_decimal, lon_decimal, address_str remain None as initialized.
        else:  # This else corresponds to `if exif_data:`
            logger.warning(
                "No EXIF data found in the image. Skipping all EXIF-dependent processing."
            )
            # vienna_datetime_iso, address_str, lat_decimal, lon_decimal remain None as initialized.

        # Number Plate Extraction with Gemini
        number_plate_str = None
        if GOOGLE_API_KEY:
            try:
                # Ensure the image stream is reset if it was read before for Pillow
                img_stream.seek(0)
                image_bytes = img_stream.read()

                response = generate(image_bytes, GOOGLE_API_KEY)

                extracted_text = response.text.strip() if response.text else ""
                if (
                    extracted_text
                    and extracted_text.upper() != GEMINI_NO_PLATE_RESPONSE.upper()
                ):
                    number_plate_str = extracted_text
                else:
                    number_plate_str = (
                        None  # Default to None for N/A, empty, or whitespace-only
                    )
                    if not response.text:
                        logger.warning(
                            "Gemini response for number plate was empty or had no text part."
                        )
                    elif (
                        not extracted_text
                    ):  # response.text was not empty but became empty after strip
                        logger.warning(
                            "Gemini response for number plate consisted of only whitespace."
                        )
                    elif extracted_text.upper() == GEMINI_NO_PLATE_RESPONSE.upper():
                        # Optionally log this case, or just let it be None silently
                        logger.info(
                            f"Gemini reported '{GEMINI_NO_PLATE_RESPONSE}' for number plate."
                        )
            except Exception as e:
                logger.error(
                    f"Error during number plate extraction with Gemini: {str(e)}"
                )
                number_plate_str = None  # Ensure it's None on error
        else:
            logger.info(
                "Skipping number plate extraction as GOOGLE_API_KEY is not set."
            )

        # Prepare response data
        response_payload = {
            "filename": file.filename,
        }
        if vienna_datetime_iso:
            response_payload["datetime_original_vienna"] = vienna_datetime_iso
        if lat_decimal is not None and lon_decimal is not None:
            response_payload["latitude"] = lat_decimal
            response_payload["longitude"] = lon_decimal
        if address_str:  # Only add address if it was successfully retrieved
            response_payload["address"] = address_str
        if number_plate_str:  # Only add number plate if it was successfully extracted
            response_payload["number_plate"] = number_plate_str

        return response_payload

        # ValueError from strptime is now handled inline within the EXIF processing block.
        # The generic Exception handler remains.
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in extract_info: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred.",  # Avoid leaking raw error details
        )


def generate(image_data: bytes, api_key: str) -> types.GenerateContentResponse:
    client = genai.Client(
        api_key=api_key,
    )

    # Use constants for model name and prompt
    model = GEMINI_MODEL_NAME
    prompt = GEMINI_PROMPT

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
