
from datetime import datetime



class Settings:
    def __init__(self, first_timer, second_timer, pellets, seconds, confidence):
        self.__first_timer = first_timer
        self.__second_timer = second_timer
        self.__pellets = pellets
        self.__seconds = seconds
        self.__confidence = confidence

    # Getter methods
    def get_first_timer(self):
        return self.__first_timer

    def get_second_timer(self):
        return self.__second_timer

    def get_pellets(self):
        return self.__pellets

    def get_seconds(self):
        return self.__seconds

    def get_confidence(self):
        return self.__confidence

    # Setter methods
    def set_first_timer(self, first_timer):
        self.__first_timer = first_timer

    def set_second_timer(self, second_timer):
        self.__second_timer = second_timer

    def set_pellets(self, pellets):
        self.__pellets = pellets

    def set_seconds(self, seconds):
        self.__seconds = seconds

    def set_confidence(self, confidence):
        self.__confidence = confidence

class Line_Chart_Data():
    def __init__(self, date, TimeRecord):
        self.__date = date
        self.__TimeRecord = TimeRecord

    def set_date(self, date):
        self.__date = date

    def set_timeRecord(self, timeRecord):
        self.__TimeRecord = timeRecord

    def get_date(self):
        if isinstance(self.__date, str):
            date_obj = datetime.strptime(self.__date, "%Y-%m-%d")
        else:
                date_obj = self.__date

        # Format the datetime object as a string
        return date_obj.strftime("%Y-%m-%d")

    def get_timeRecord(self):
        return self.__TimeRecord


class Email():
    def __init__(self, sender_email, recipient_email, APPPassword, days):
        self.__sender_email = sender_email
        self.__recipient_email = recipient_email
        self.__APPPassword = APPPassword
        self.__days = days

    # Accessor method for sender_email
    def get_sender_email(self):
        return self.__sender_email

    # Mutator method for sender_email
    def set_sender_email(self, new_sender_email):
        self.__sender_email = new_sender_email

    # Accessor method for recipient_email
    def get_recipient_email(self):
        return self.__recipient_email

    # Mutator method for recipient_email
    def set_recipient_email(self, new_recipient_email):
        self.__recipient_email = new_recipient_email

    # Accessor method for APPPassword
    def get_APPPassword(self):
        return self.__APPPassword

    # Mutator method for APPPassword
    def set_APPPassword(self, new_APPPassword):
        self.__APPPassword = new_APPPassword

    # Accessor method for days
    def get_days(self):
        return self.__days

    # Mutator method for days
    def set_days(self, new_days):
        self.__days = new_days


class Line_Chart_Data_Pellets():
    def __init__(self, date, pellets):
        self.__date = date
        self.__pellets = pellets

    def set_date(self, date):
        self.__date = date

    def set_pellets(self, pellets):
        self.__pellets = pellets

    def get_date(self):
        return self.__date.strftime("%Y-%m-%d")

    def get_pellets(self):
        return self.__pellets