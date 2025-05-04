import os
import json
import gspread
from datetime import datetime
from google.oauth2.service_account import Credentials

def log_to_google_sheet(sheet_name, row_data, worksheet_title="Sheet1"):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]

    # Load credentials from environment variable
    creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not creds_json:
        print("Google Sheets logging skipped. Missing credentials.")
        return

    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)

    client = gspread.authorize(creds)
    sheet = client.open(sheet_name).worksheet(worksheet_title)
    sheet.append_row(row_data)
