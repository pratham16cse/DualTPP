import time
from datetime import datetime

def getDatetime(epoch):
	date_string=time.strftime("%d-%m-%Y %H:%M:%S", time.localtime(epoch))
	date= datetime.strptime(date_string, "%d-%m-%Y %H:%M:%S")
	return date

def isWeekend(epoch):
	date=getDatetime(epoch)
        #print(date, date.weekday())
	if date.weekday() in (5,6):
		return 1
	else:
		return 0

def minutesOfHour(epoch):
	date=getDatetime(epoch)
	oneHot=[0]*60
	oneHot[date.minute]=1
	return oneHot

def hourOfDay(epoch):
	date=getDatetime(epoch)
	oneHot=[0]*24
	oneHot[date.hour]=1
	return oneHot

def dayOfWeek(epoch):
	date=getDatetime(epoch)
	oneHot=[0]*7
	oneHot[date.weekday()]=1
	return oneHot

def dayOfMonth(epoch):
	date=getDatetime(epoch)
	oneHot=[0]*31
	oneHot[date.day-1]=1
	return oneHot

def monthOfYear(epoch):
	date=getDatetime(epoch)
	oneHot=[0]*12
	oneHot[date.month-1]=1
	return oneHot

def gap(epoch1, epoch2):
	diff=epoch2-epoch1
        #print(epoch1, epoch2, diff)
	if diff >= 0:
		return diff/3600.0
	else :
		return diff*(-1.0)/3600.0

def secondOfFeatures(epoch):
    '''
    [sec_of_min, sec_of_hr, sec_of_day, sec_of_week, sec_of_month,
        sec_of_year]
    '''
    date = getDatetime(epoch)
    ts_mi = time.mktime(datetime(date.year, date.month, date.day, date.hour, date.minute, 0).timetuple())
    ts_h = time.mktime(datetime(date.year, date.month, date.day, date.hour, 0, 0).timetuple())
    ts_d = time.mktime(datetime(date.year, date.month, date.day, 0, 0, 0).timetuple())
    ts_w = ts_d - date.weekday()*(60.0*60.0*24.0)
    ts_mo = time.mktime(datetime(date.year, date.month, 1, 0, 0, 0).timetuple())
    ts_y = time.mktime(datetime(date.year, 1, 1, 0, 0, 0).timetuple())
#    featVec = [epoch-i for i in [ts_mi, ts_h, ts_d, ts_w, ts_mo, ts_y]]
#    featVec = [i for i in [epoch-ts_mi, ts_mi-ts_h, ts_h-ts_d, ts_d-ts_w, ts_d-ts_mo, ts_mo-ts_y]]
#    featVec = [i for i in [epoch-ts_mi, ts_mi-ts_h, ts_h-ts_d, ts_d-ts_w, ts_d-ts_mo]]
#    featVec = [date.second, date.minute, date.hour, date.day, date.month, date.year]
    featVec = [date.second, date.minute, date.hour, date.day, date.month]
#    featVec = [date.hour, date.day, date.month]
    return featVec
#print(dayOfWeek(1522635973))
