import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

def log_to_google_sheet(sheet_name, row_data, worksheet_title="Sheet1", creds_path="config/google_credentials.json"):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    client = gspread.authorize(creds)
    sheet = client.open(sheet_name).worksheet(worksheet_title)
    sheet.append_row(row_data)
