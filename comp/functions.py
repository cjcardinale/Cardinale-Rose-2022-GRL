import numpy as np
import xarray as xr
import scipy.integrate as integ
import warnings
#import scipy.stats as sp
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
    return xr.apply_ufunc(
        function, array,
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
        
#def integrate_old(array):
#    """Computes the vertical integral (trapezoidal Rule)."""
#    return (np.squeeze(integ.trapz(array,x=(TEMP_rp.plev)/g,axis=-1)))

#def integrate_trop_old(array):
#    """Computes the vertical integral over the troposphere (1000-300 hPa)"""
#    return (np.squeeze(integ.trapz(array[:,10:28],x=(TEMP_rp.plev[10:28])/g,axis=-1)))

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

def detrend_gufunc(array):
    return signal.detrend(array,0)
def detrend(array):
    """Linearly detrend array"""
    try:
        dt = xr.apply_ufunc(
            detrend_gufunc, array,
            dask='allowed',
            output_dtypes=[float])
        print('no nans found')
    except ValueError: 
        dt = xr.apply_ufunc(
            detrend_gufunc, array.ffill('time',limit=80).bfill('time',limit=80),
            dask='allowed',
            output_dtypes=[float])
        print('nans found, array filled using ffill and bfill')
    return dt

def anom_daily_filt(array,period=None):
    """Method for calculating anomalies from a short array by first computing the 12-day rolling-mean.
    It is possible to construct climatologies by averaging all ensemble members; however, this introduces 
    errors in calculating efficiency—e.g.,the annual average anomaly would be non-zero for each member."""
    gb = array.groupby('time.dayofyear')
    if gb.mean().dayofyear.size == 214:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        #first create an array with all days
        time = xr.cftime_range(str(array.time.dt.date[0].values),str(array.time.dt.date[-1].values),freq='D',calendar='noleap').to_datetimeindex()
        full_array = xr.DataArray(data=np.arange(time.size),coords={'time': time})
        gb2 = (xr.broadcast(full_array,array)[1].ffill('time',limit=80).bfill('time',limit=80)
            ).rolling(center=True,min_periods=1,time=12).mean().sel(time=period).groupby('time.dayofyear')
    if gb.mean().dayofyear.size == 366:
        gb2 = array.rolling(center=True,min_periods=1,time=12).mean().groupby('time.dayofyear')
    clim = gb2.mean(['time'])
    return (gb - clim).drop('dayofyear')

def time_tendency(array):
    return array.differentiate('time',datetime_unit='D',edge_order=2)

def low_filt(data,N=1,Wn=1/2,btype='low'):
    """2-day low-pass filter"""
    #B, A = signal.butter(N, Wn, btype, output='ba')
    #filt = signal.filtfilt(B,A, data)
    sos = signal.butter(N, Wn, btype, output='sos')
    filt = signal.sosfiltfilt(sos, data)
    return filt

