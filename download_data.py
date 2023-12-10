import cdsapi

m=[]
for i in range(1,12+1):
    if(i<10):m.append('0'+str(i))
    else:m.append(str(i))

d=[]
for i in range(1,31+1):
    if(i<10):d.append('0'+str(i))
    else:d.append(str(i))

print(m,d)

def download(year):
    c = cdsapi.Client()
    save_path='/home/ices/dataset/ERA5_MV/precipitation/'+year+'.grib'

    c.retrieve(
        'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': 'relative_humidity',
        'year': [year],
        'month': m,
        'day': d,
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            40, 110, 20, 130,
        ],
            'format': 'grib',
            'grid': [0.25, 0.25]
        },
        save_path)

download(year='2018')