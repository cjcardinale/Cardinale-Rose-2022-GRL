import numpy as np
import xarray as xr
import pandas as pd
import scipy.integrate as integ
import warnings
import scipy.stats as sp
from scipy import signal

C = 40075000. * np.cos(np.deg2rad(70.))
A = (2.*6371000.**2*np.pi)*(np.sin(np.deg2rad(90.))-np.sin(np.deg2rad(70.))) 
sid = 86400.
g = 9.81
Lv = 2.5e6
Cp = 1004.
Rd = 287

def h(circ=C,area=A):
    """conversion factor:
    energy flux to local MSE flux convergence (W/m2/100 hPa)"""
    return (circ/(area*g))*10**4

def pc_average(ds):
    """computes the polar cap-average or any latitude-longitude average"""
    lon=ds.lon
    lat=ds.lat
    lat_field = xr.broadcast(lat,lon)[0]
    lat_field = (lat_field*(ds.isnull()==False)).where((lat_field*(ds.isnull()==False))>0)
    weights=np.cos(np.deg2rad(lat_field))/np.cos(np.deg2rad(lat_field)).mean('lat')
    avg = (ds*weights).mean(['lon','lat'])
    return avg

def bracket(array):
    """zonal mean"""
    return array.mean(dim='lon')
        
def bracket2(array):
    """Alternate zonal mean function, which accounts for topography without replacing nans."""
    return xr.DataArray(np.squeeze(integ.trapz(array.to_masked_array(),x=array.lon,axis=-1)) / 359.375
                ,coords=[array.time,array.plev],dims=['time','plev'])

def bar(array):
    """time mean"""
    return array.mean(dim='time')

def star(array):
    """departure fomr the zonal mean"""
    return array - bracket(array)
        
def ufunc(array,function):
    """Use with integrate and curly_bracket functions
    to preserve the labeled coordinates and dimensions of the DataArray."""
    return xr.apply_ufunc(
        function, array,
        input_core_dims=[['plev']],
        vectorize=True,
        exclude_dims={'plev'},
        dask='allowed',
        output_dtypes=[float])

def ufunc2(array,function):
    ax=array.get_axis_num('time')
    return xr.apply_ufunc(
        function, array, ax,
        dask='allowed',
        output_dtypes=[float])

def curly_bracket(array):
    """Computes the mass weighted vertical average.
    Note that the integrate function has changed in the latest SciPy release."""
    return (array.integrate('plev')/g) / ((100000. / g))
        
def double_prime(array):
    """departure from the mass weighted vertical average"""
    return array - curly_bracket(array)

def integrate(array):
    """Computes the vertical integral (trapezoidal Rule)."""
    return (array.integrate('plev')/g)

def integrate_trop(array):
    """Computes the vertical integral over the troposphere (1000-300 hPa)"""
    return (array.sel(plev=slice(30000,100000)).integrate('plev')/g)

#if xarray is not used, use scipy's integrate function
#plev = np.array([   500,    700,   1000,   2000,   3000,   4000,   5000,   7000,  10000,
#        20000,  30000,  40000,  50000,  55000,  60000,  65000,  70000,  72500,
#        75000,  77500,  82500,  85000,  87500,  90000,  92500,  95000,  97500,
#       100000])
#def integrate_old(array):
#    """Computes the vertical integral (trapezoidal Rule)."""
#    return (np.squeeze(integ.trapz(array,x=plev/g,axis=-1)))
#def integrate_trop_old(array):
#    """Computes the vertical integral over the troposphere (1000-300 hPa)"""
#    return (np.squeeze(integ.trapz(array[:,:,10:28],x=plev[10:28]/g,axis=-1)))

def eddy(wind,energy):
    return bracket(((star(wind.fillna(0))) * (star(energy.fillna(0)))))
        
def eddy_zonal(wind,energy):
    """Use this function for the zonal struction of the eddy flux and the contribution of 
    the eddy flux to the MSE flux."""
    return (star(wind) * star(energy)).fillna(0).drop('lat')

def mmc(wind,energy):
    return (double_prime(bracket(wind.fillna(0))) 
            * double_prime(bracket(energy.fillna(0))))

def daily_mean(array):
    return array.groupby(array.time.dt.floor('D')).mean('time').rename({'floor':'time'})

def mse_flx(wind,energy):
    """Method used for MSE calculation in the paper. Computes both the eddy and mean meridional circulation fluxes,
    neglecting contributions from the net mass flux."""
    eddy_flux = eddy(wind,energy)
    mmc_flux = mmc(wind,energy)
    mse = eddy_flux + mmc_flux
    return mse.drop('lat')
        
def mse_liang(wind,energy,pc_avg_energy):
    """Method from Liang et al. 2018 (https://doi.org/10.1007/s00382-017-3722-x.) 
    You can assume the winter and polar cap-averaged MSE to be about 300000 J/m2, but for
    for increased accuracy, download the polar cap data poleward of 70N and apply the following 
    area (average function) and vertical (curly_bracket function) averaging:
            
    def average(ds):
        lon=ds.lon
        lat=ds.lat
        lat_field = xr.broadcast(lat,lon)[0]
        lat_field = (lat_field*(ds.isnull()==False)).where((lat_field*(ds.isnull()==False))>0)
        weights=np.cos(np.deg2rad(lat_field))/np.cos(np.deg2rad(lat_field)).mean('lat')
        avg = (ds*weights).mean(['lon','lat'])
        return avg
            
    pc_avg_energy=curly_bracket(average(merra.TEMP*Cp+merra.H*g+merra.QV*Lv))
    """
    return (bracket((wind * (energy-pc_avg_energy)).fillna(0)))

#apply detrending function to the time dimension
def detrend_gufunc(array,ax):
    return signal.detrend(array,axis=ax)
def detrend(array):
    """Linearly detrend array"""
    ax=array.get_axis_num('time')
    try:
        dt = xr.apply_ufunc(
            detrend_gufunc, array, ax,
            dask='allowed',
            output_dtypes=[float])
        print('no nans found')
    except ValueError: 
        dt = xr.apply_ufunc(
            detrend_gufunc, array.ffill('time',limit=80).bfill('time',limit=80), ax,
            dask='allowed',
            output_dtypes=[float])
        print('nans found, array filled using ffill and bfill')
    return dt

def low_filt(data,ax,N=1,Wn=1/2,btype='low'):
    """2-day low-pass filter
    set axis to time"""
    #B, A = signal.butter(N, Wn, btype, output='ba')
    #filt = signal.filtfilt(B,A, data)
    sos = signal.butter(N, Wn, btype, output='sos')
    filt = signal.sosfiltfilt(sos, data, axis=ax)
    return filt

def low_filt_12(data,ax,N=1,Wn=1/12,btype='low'):
    """12-day low-pass filter"""
    #B, A = signal.butter(N, Wn, btype, output='ba')
    #filt = signal.filtfilt(B,A, data)
    sos = signal.butter(N, Wn, btype, output='sos')
    filt = signal.sosfiltfilt(sos, data, axis=ax)
    return filt

#need to add nan values to arrays for proper rolling means (don't include dates in october for april rolling means, for example)
#we use rolling means, but we can just use a 12 day low pass filter as long as the sosfiltfilt axis is set to 'time' (usually 1)
def anom_daily_filt(array,period=None):
    """Method for calculating anomalies from a short array by first computing the 12-day rolling-mean.
    It is possible to construct climatologies by averaging all ensemble members; however, this introduces 
    errors in calculating efficiency—e.g.,the annual average anomaly would be non-zero for each member."""
    gb = array.groupby('time.dayofyear')
    if gb.mean().dayofyear.size == 214:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        #first create an array with all days
        time = xr.cftime_range(str(array.time.dt.date[0].values),str(array.time.dt.date[-1].values)
                              ,freq='D',calendar='noleap').to_datetimeindex()
        full_array = xr.DataArray(data=np.arange(time.size),coords={'time': time})
        #compute rolling means over the period, with nans added for other days
        gb2 = (xr.broadcast(full_array,array)[1]
            ).rolling(center=True,min_periods=1,time=12).mean().sel(time=period).groupby('time.dayofyear')
    if gb.mean().dayofyear.size == 366:
        gb2 = array.rolling(center=True,min_periods=1,time=12).mean().groupby('time.dayofyear')
    clim = gb2.mean(['time'])
    return (gb - clim).drop('dayofyear')

def anom_daily_filt2(array):
    """Method for calculating anomalies from a short array by first using a 12-day low pass filter.
    It is possible to construct climatologies by averaging all ensemble members; however, this introduces 
    errors in calculating efficiency—e.g.,the annual average anomaly would be non-zero for each member."""
    gb = array.groupby('time.dayofyear')
    gb2 = ufunc2(array,low_filt_12).groupby('time.dayofyear')
    clim = gb2.mean(['time'])
    return (gb - clim).drop('dayofyear')

def clim(array,period=None):
    """Method for calculating climatologies from a short array by first computing the 12-day rolling-mean.
    It is possible to construct climatologies by averaging all ensemble members; however, this introduces 
    errors in calculating efficiency—e.g.,the annual average anomaly would be non-zero for each member."""
    gb = array.groupby('time.dayofyear')
    if gb.mean().dayofyear.size == 214:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        #first create an array with all days
        time = xr.cftime_range(str(array.time.dt.date[0].values),str(array.time.dt.date[-1].values),freq='D',calendar='noleap').to_datetimeindex()
        #compute rolling means over the period, with nans added for other days
        full_array = xr.DataArray(data=np.arange(time.size),coords={'time': time})
        gb2 = (xr.broadcast(full_array,array)[1]
            ).rolling(center=True,min_periods=1,time=12).mean().sel(time=period).groupby('time.dayofyear')
    if gb.mean().dayofyear.size == 366:
        gb2 = array.rolling(center=True,min_periods=1,time=12).mean().groupby('time.dayofyear')
    clim = gb2.mean(['time'])
    return clim

def clim2(array):
    """Method for calculating climatologies from a short array by first using a 12-day low pass filter.
    It is possible to construct climatologies by averaging all ensemble members; however, this introduces 
    errors in calculating efficiency—e.g.,the annual average anomaly would be non-zero for each member."""
    gb = array.groupby('time.dayofyear')
    gb2 = ufunc2(array,low_filt_12).groupby('time.dayofyear')
    clim = gb2.mean(['time'])
    return clim

def clim3(array):
    """Method for calculating climatologies by averaging all ensembles."""
    gb = array.groupby('time.dayofyear')
    clim = gb.mean(['time','member_id'])
    return clim

def time_tendency(array):
    return array.differentiate('time',datetime_unit='D',edge_order=2)

def composite(array,events):
    """Daily composite in the 30 days before and after an event date"""
    comp = xr.concat([array.sel(time=slice(array.time.shift(time=30).sel(time=events[x])
                                           ,array.time.shift(time=-30).sel(time=events[x]))).drop('time')
                      for x in range(len(events))],'event').mean('event')
    return comp

def composite_event(array,events):
    """Similar to the composite function, without averaging over each event. Helpful when
    computing the standard deviation or performing certain significance testing."""
    comp = xr.concat([array.sel(time=slice(array.time.shift(time=30).sel(time=events[x])
                                           ,array.time.shift(time=-30).sel(time=events[x]))).drop('time')
                      for x in range(len(events))],'event')
    comp = comp.assign_coords({'event': comp.event})
    return comp

def mci(data, confidence=0.95):
    """method for calculating the mean confidence interval"""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.sem(a)
    h = se * sp.t.ppf((1 + confidence) / 2., n-1)
    return h

############################################################################################################
# method for identifying tropospheric energy flux events 

def event_days(sigma,array):
    """Returns days above a threshold. Sigma=0 for this particular method."""
    SD = array.std('time')
    high_MSE_time = xr.where(array>(SD*sigma)
         ,array,np.nan)
    eff_high_MSE_time = high_MSE_time.dropna('time').time
    return eff_high_MSE_time

def event_average(array,events):
    "Returns a cumulative average of each event. The 'event array' should include all consecutive days in each event. The cumulative average of the event will be returned on the start day of the event."
    df = (xr.broadcast(array.sel(time=events),array.time)[0]).to_dataframe('df')
    df2 = (xr.broadcast(array.sel(time=events).where(np.logical_and(array>=0,array<=1)),array.time)[0]).to_dataframe('df2')
    df3 = (xr.broadcast(array.sel(time=events),array.time)[0]).to_dataframe('df3')
    df_avg = df2['df2'].groupby(df['df'].isna().cumsum()).expanding().mean().reset_index()
    df_avg2 = df_avg.groupby('df').first().dropna()
    df_avg3 = df_avg.groupby('df').last().dropna()
    bin_size = df3['df3'].groupby(df['df'].isna().cumsum()).expanding().count().reset_index()
    bin_size2 = bin_size.replace(0,np.nan)
    bin_size3 = bin_size2.groupby('df').first().dropna()
    bin_size4 = bin_size2.groupby('df').last().dropna()
    
    start = np.sort(np.concatenate([df_avg2.where(np.logical_and(df_avg2.time.dt.day==31,df_avg2.time.dt.month==3)).dropna().time.to_numpy()+np.array(215, dtype='timedelta64[D]'),
              df_avg2.where(np.logical_or(df_avg2.time.dt.day!=31,df_avg2.time.dt.month!=3)).dropna().time.to_numpy()
             +np.array(1, dtype='timedelta64[D]')]))
    
    start2 = np.sort(np.concatenate([bin_size3.where(np.logical_and(bin_size3.time.dt.day==31,bin_size3.time.dt.month==3)).dropna().time.to_numpy()+np.array(215, dtype='timedelta64[D]'),
              bin_size3.where(np.logical_or(bin_size3.time.dt.day!=31,bin_size3.time.dt.month!=3)).dropna().time.to_numpy()
             +np.array(1, dtype='timedelta64[D]')]))
    
    end = df_avg3['df2']
    end2 = bin_size4['df3']
    
    c_size = xr.DataArray(end2.values,coords=[start2],dims=['time'])
    c_avg = xr.DataArray(df_avg['df2'].values,coords=[df_avg['time'].values],dims=['time'])
    return c_avg,c_size


def event_integ(array,events):
    "Returns a cumulative average of each event. The 'event array' should include all consecutive days in each event. The cumulative average of the event will be returned on the start day of the event."
    df = (xr.broadcast(array.sel(time=events),array.time)[0]).to_dataframe('df')
    df2 = (xr.broadcast(array.sel(time=events),array.time)[0]).to_dataframe('df2')
    df_avg = (df2['df2'].fillna(0).groupby(df['df'].isna().cumsum()).expanding()
    .apply(integ.trapz,kwargs={'dx':sid}).replace(0,np.nan).reset_index())
    
    events = pd.concat([df_avg.where(df_avg['df2'] >x*10**6).groupby('df').first() 
                for x in np.arange(8,112,8)]).sort_values(by=['time'])
    
    c_integ = xr.DataArray(events['df2'].values*10**-6,coords=[events.time],dims=['time'])
    c_integ_full = xr.DataArray(df_avg['df2'].values*10**-6,coords=[df_avg['time'].values],dims=['time'])
    return c_integ,c_integ_full
