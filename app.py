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
from typing import Tuple, Any, Optional, Protocol
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
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat_decimal}&lon={lon_decimal}&accept-language=de"
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
EXIF_IFD_POINTER_TAG = 34665  # Tag ID for ExifOffset, points to the ExifIFD


class RationalLike(Protocol):
    numerator: int
    denominator: int


def get_decimal_from_dms(dms: Tuple[RationalLike, RationalLike, RationalLike], ref: str) -> float:
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


async def _load_image_from_upload(file: UploadFile) -> Tuple[Image.Image, io.BytesIO]:
    """Loads image from upload, validates, and returns PIL Image and BytesIO stream."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    if not file.filename.lower().endswith((".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="File is not a JPG image")

    contents = b""
    try:
        contents = await file.read()
        img_stream = io.BytesIO(contents)
        img = Image.open(img_stream)
        img_stream.seek(0)  # Reset stream for potential re-read
        return img, img_stream
    except UnidentifiedImageError:
        logger.warning("Cannot identify image file. It might be corrupted or not a valid JPG.")
        raise HTTPException(
            status_code=400,
            detail="Cannot identify image file. It might be corrupted or not a valid JPG.",
        )
    except Exception as e:
        logger.error(f"Could not open or read image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Could not open image: {str(e)}")
    finally:
        await file.close()


def _get_datetime_original(exif_ifd: Optional[dict]) -> Optional[str]:
    """Extracts and formats DateTimeOriginal from EXIF IFD."""
    if not exif_ifd:
        logger.info("ExifIFD not found. Skipping datetime processing.")
        return None

    datetime_original_str = exif_ifd.get(EXIF_DATETIME_ORIGINAL_TAG)
    if not datetime_original_str:
        logger.warning("EXIF DateTimeOriginal tag not found. Skipping datetime processing.")
        return None

    try:
        naive_datetime = datetime.strptime(datetime_original_str, "%Y:%m:%d %H:%M:%S")
        vienna_datetime = VIENNA_TZ.localize(naive_datetime)
        return vienna_datetime.isoformat()
    except ValueError:
        logger.warning(
            f"Invalid datetime format in EXIF data: '{datetime_original_str}'. Skipping datetime processing."
        )
        return None


async def _get_location_data_from_gps(
    raw_gps_info: Optional[dict],
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Extracts GPS coordinates from EXIF and fetches address via Nominatim."""
    lat_decimal: Optional[float] = None
    lon_decimal: Optional[float] = None
    address_str: Optional[str] = None

    if not raw_gps_info:
        logger.info("GPS metadata (tag GPS_INFO_TAG_ID) not found in EXIF. Skipping GPS processing.")
        return None, None, None

    decoded_gps_info = {
        GPSTAGS.get(tag_id, tag_id): value for tag_id, value in raw_gps_info.items()
    }

    gps_latitude = decoded_gps_info.get("GPSLatitude")
    gps_latitude_ref = decoded_gps_info.get("GPSLatitudeRef")
    gps_longitude = decoded_gps_info.get("GPSLongitude")
    gps_longitude_ref = decoded_gps_info.get("GPSLongitudeRef")

    if not (gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref):
        logger.warning("Incomplete GPS data in EXIF. Skipping GPS processing.")
        return None, None, None

    try:
        lat_decimal = get_decimal_from_dms(gps_latitude, gps_latitude_ref)
        lon_decimal = get_decimal_from_dms(gps_longitude, gps_longitude_ref)
    except (TypeError, ZeroDivisionError, IndexError) as e:
        logger.error(f"Error converting DMS to decimal: {e}")
        return None, None, None

    if lat_decimal is not None and lon_decimal is not None:
        nominatim_url = NOMINATIM_URL.format(lat_decimal=lat_decimal, lon_decimal=lon_decimal)
        headers = {"User-Agent": NOMINATIM_USER_AGENT}
        try:
            async with httpx.AsyncClient() as client:
                api_response = await client.get(nominatim_url, headers=headers, timeout=10.0)
            api_response.raise_for_status()
            address_data = api_response.json()
            address_components = address_data.get("address", {})

            address_parts = []
            street = address_components.get("road")
            if street:
                house_number = address_components.get("house_number")
                full_street = f"{street} {house_number}" if house_number else street
                address_parts.append(full_street)
            postcode = address_components.get("postcode")
            if postcode:
                address_parts.append(postcode)
            city_keys = ["city", "town", "village", "hamlet"]
            city_name = next(
                (address_components.get(key) for key in city_keys if address_components.get(key)),
                None,
            )
            if city_name:
                address_parts.append(city_name)
            if address_parts:
                address_str = ", ".join(address_parts)
        except httpx.RequestError as exc:
            logger.error(f"Nominatim request error for {exc.request.url!r}: {exc}")
        except httpx.HTTPStatusError as exc:
            logger.error(f"Nominatim API error {exc.response.status_code} for {exc.request.url!r}.")
        except Exception as e:
            logger.error(f"Unexpected error during geocoding: {str(e)}")
            # address_str remains None as initialized

    return lat_decimal, lon_decimal, address_str


async def _process_exif_data(
    img: Image.Image,
) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[str]]:
    """Processes EXIF data from the image to extract datetime, GPS, and address."""
    vienna_datetime_iso: Optional[str] = None
    lat_decimal: Optional[float] = None
    lon_decimal: Optional[float] = None
    address_str: Optional[str] = None

    exif_data = img.getexif()
    if not exif_data:
        logger.warning("No EXIF data found. Skipping all EXIF-dependent processing.")
        return None, None, None, None

    logger.info("EXIF data found. Processing...")
    exif_ifd = exif_data.get_ifd(EXIF_IFD_POINTER_TAG)
    vienna_datetime_iso = _get_datetime_original(exif_ifd)

    raw_gps_info = exif_data.get_ifd(GPS_INFO_TAG_ID)
    lat_decimal, lon_decimal, address_str = await _get_location_data_from_gps(raw_gps_info)

    return vienna_datetime_iso, lat_decimal, lon_decimal, address_str


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
    try:
        img, img_stream = await _load_image_from_upload(file)

        # Process EXIF data
        vienna_datetime_iso, lat_decimal, lon_decimal, address_str = await _process_exif_data(img)

        # Number Plate Extraction
        image_bytes = img_stream.getvalue()
        number_plate_str = _extract_number_plate_from_image_data(image_bytes, GOOGLE_API_KEY)

        # Prepare response data
        response_payload = {"filename": file.filename} # file.filename is safe due to _load_image_from_upload validation
        if vienna_datetime_iso:
            response_payload["datetime_original_vienna"] = vienna_datetime_iso
        if lat_decimal is not None and lon_decimal is not None:
            response_payload["latitude"] = lat_decimal
            response_payload["longitude"] = lon_decimal
        if address_str:
            response_payload["address"] = address_str
        if number_plate_str:
            response_payload["number_plate"] = number_plate_str

        return response_payload

    except HTTPException:  # Re-raise HTTPExceptions directly as they are already well-defined
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in extract_info endpoint: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred.",
        )


def _call_gemini_for_number_plate(
    image_data: bytes, api_key: str
) -> types.GenerateContentResponse:
    """Calls the Gemini API to analyze an image for number plates."""
    client = genai.Client(api_key=api_key)
    model = GEMINI_MODEL_NAME
    prompt = GEMINI_PROMPT
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(mime_type="image/jpeg", data=image_data),
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="text/plain",
    )
    return client.models.generate_content(
        model=model, contents=contents, config=generate_content_config
    )


def _extract_number_plate_from_image_data(
    image_bytes: bytes, api_key: Optional[str]
) -> Optional[str]:
    """Extracts number plate from image bytes using Gemini API."""
    if not api_key:
        logger.info("Skipping number plate extraction as GOOGLE_API_KEY is not set.")
        return None

    try:
        response = _call_gemini_for_number_plate(image_bytes, api_key)
        extracted_text = response.text.strip() if response.text else ""

        if extracted_text and extracted_text.upper() != GEMINI_NO_PLATE_RESPONSE.upper():
            return extracted_text

        # Log reasons for not returning a plate if it's N/A or empty
        if not response.text:
            logger.warning("Gemini response for number plate was empty or had no text part.")
        elif not extracted_text:
            logger.warning("Gemini response for number plate consisted of only whitespace.")
        elif extracted_text.upper() == GEMINI_NO_PLATE_RESPONSE.upper():
            logger.info(f"Gemini reported '{GEMINI_NO_PLATE_RESPONSE}' for number plate.")
        return None # Default to None for N/A, empty, or whitespace-only response

    except Exception as e:
        logger.error(f"Error during number plate extraction with Gemini: {str(e)}")
        return None
