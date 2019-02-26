select date_format(Date,'%Y %m %d %H'), count(date_format(Date,'%Y %m %d %H')) from Demographic group by date_format(Date,'%Y %m %d %H');
