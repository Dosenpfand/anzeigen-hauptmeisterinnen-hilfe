# FalschparkerApp

## Description

FalschparkerApp is a web application specifically designed to assist users in Vienna, Austria, with reporting wrongly parked vehicles. It extracts information from images of vehicles, focusing on identifying license plates and retrieving EXIF data such as GPS location and the time the photo was taken.

The application uses Google's Gemini AI to analyze images for license plates and Nominatim for reverse geocoding GPS coordinates. The timezone for EXIF data is set to Vienna ("Europe/Vienna").

## Features

*   **Image Upload**: Users can upload images of vehicles.
*   **EXIF Data Extraction**: Extracts date, time, and GPS coordinates (if available) from image metadata.
*   **License Plate Recognition**: Utilizes Google Gemini AI to identify vehicle license plates from the image.
*   **Location Lookup**: Converts GPS coordinates to a human-readable address using Nominatim.
*   **Rate Limiting**: Implements rate limiting to prevent abuse.

## Getting Started

### Prerequisites

*   Docker and Docker Compose
*   A Google API Key with access to the Gemini API.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your Google API key:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    ```

3.  **Build and run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```
    The application will be available at `http://localhost:8000`.

## Usage

1.  Navigate to `http://localhost:8000` in your web browser.
2.  Upload an image of a vehicle.
3.  The application will process the image and display the extracted license plate, date, time, and location (if available).

The API endpoint for extracting information is `/extract_info/`.

## Configuration

The application can be configured using environment variables:

*   `GOOGLE_API_KEY`: Your Google API Key for Gemini.
*   `RATE_LIMIT_REQUESTS`: Maximum number of requests allowed within the rate limit window. (Default: 20)
*   `RATE_LIMIT_HOURS`: The duration of the rate limit window in hours. (Default: 10)
*   `NOMINATIM_USER_AGENT`: User agent string for Nominatim API requests. (Default: "FalschparkerApp/0.1 (falschparker@sad.bz)")

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the [LICENSE](./LICENSE) file.
