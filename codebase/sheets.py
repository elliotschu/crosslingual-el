"""
Elliot Schumacher, Johns Hopkins University
Created 3/1/19
"""

"""
Elliot Schumacher, Johns Hopkins University
Created 7/17/18

https://stackoverflow.com/questions/46274040/append-a-list-in-google-sheet-from-python
https://developers.google.com/sheets/api/quickstart/python
https://stackoverflow.com/questions/36061433/how-to-do-i-locate-a-google-spreadsheet-id
"""
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import traceback
import logging
import os
import pwd
import socket

class Sheets(object):
    def __init__(self):
        self.exclude_hosts = ["amsterdam"]

        self.log = logging.getLogger()

        self.ignore_updates = self.is_excluded()

    def is_excluded(self):
        is_excluded = False
        if self.get_credentials() is None:
            is_excluded = True
            self.log.info("No credentials found")
        else:
            host = socket.gethostname()
            for eh in self.exclude_hosts:
                if eh in host:
                    is_excluded = True
                    break
        return is_excluded

    def get_credentials(self):
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        credentials = None

        if os.path.exists('../client_secret.json'):
            credentials = ServiceAccountCredentials.from_json_keyfile_name('../client_secret.json', scope)
        elif os.path.exists('./client_secret.json'):
            credentials = ServiceAccountCredentials.from_json_keyfile_name('./client_secret.json', scope)

        return credentials

    def get_sheet(self, worksheet_nm = "Runs"):
        credentials = self.get_credentials()
        gc = gspread.authorize(credentials)
        wks = gc.open("CLEL-results")
        sheet = wks.worksheet(worksheet_nm)
        return sheet



    def add_run(self, results):

        if not self.ignore_updates:
            sheet = self.get_sheet()
            key_list = sheet.row_values(1)
            value_list = []

            results['args'] = str(results)
            results['status'] = 'Started'
            results['epoch'] = 0
            results['user'] = pwd.getpwuid( os.getuid() )[ 0 ]

            for i in range(len(key_list)):
                if key_list[i] in results:
                    value_list.append(results[key_list[i]])
                else:
                    value_list.append("")

            sheet.append_row(value_list)

            pass

    def update_run(self, run, end=False):
        try:
            if not self.ignore_updates:

                sheet = self.get_sheet()
                key_list = sheet.row_values(1)
                value_list = []

                if end:
                    run['status'] = 'Finished'

                current_row = sheet.find(run['timestamp'])


                for i in range(len(key_list)):
                    if key_list[i] in run:
                        sheet.update_cell(current_row.row, i+1, run[key_list[i]])
        except Exception as e:
            self.log.warning("Error writing update")
            self.log.warning(e)

    def error_run(self, run, error, end_type="Error"):
        if not self.ignore_updates:

            sheet = self.get_sheet()
            key_list = sheet.row_values(1)
            value_list = []

            run['status'] = end_type

            timesheet_cell = sheet.find(run['timestamp'])
            for i in range(len(key_list)):
                if key_list[i] in run:
                    sheet.update_cell(timesheet_cell.row, i+1, run[key_list[i]])
                if key_list[i] == 'error':
                    sheet.update_cell(timesheet_cell.row, i+1, str(error))
                if key_list[i] == 'stacktrace':
                    stacktrace = traceback.format_exc()
                    sheet.update_cell(timesheet_cell.row, i+1, stacktrace.replace("\n", "\t"))



if __name__ == "__main__":
    pass