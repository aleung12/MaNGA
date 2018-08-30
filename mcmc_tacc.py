### last major change (excludes testing): 08/28/2018
### generic initial model (no prior fit)
### fitting PA from the beginning
### fitting m=2 mode (bar mode) after 20 swap iterations
### fitting inclination angle after 40 swap iterations (03/20/2018; previously 100)
### (11/27/2017) adapt file for gas kinematic data
### (01/10/2018) adapt file for MPL-6
### (01/24/2018) modify penalty to induce smoothing of fitted parameters, see note in smoothing()
### (03/02/2018) flux-weighted convolution of kinematics
### (03/09/2018) properly deal with flux gradient at mask boundary
### (03/15/2018) corrected flux assignment for gas (previously switched)
### (08/28/2018) fixed bug in theta(phi) deprojection (corresponding changes in simdisk.py, npp.py)


from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
import numpy as np
import multiprocessing as mp
from scipy import signal
from scipy.stats import norm
import matplotlib.pyplot as plt
import time, gc, os, sys, re
import simdisk
from dirs import *


def getParams(plateid, ifudesign):
   '''
      Get values for PA and inclination from NASA-Sloan Atlas files

   '''

   ### (10/18/2017) get from fits header instead
   filename = datapath+'SPX-GAU-MILESHC/'+plateid+'/'+ifudesign+'/manga-'+plateid+'-'+ifudesign+'-MAPS-SPX-GAU-MILESHC.fits.gz'
   datafile = fits.open(filename)
   pa_deg   = round(datafile[0].header['ECOOPA'],1)
   ell      = datafile[0].header['ECOOELL']
   datafile.close()
   
   b_over_a = 1-ell
   inc_deg  = round(np.arccos(b_over_a)*180/np.pi,1)

   return pa_deg, inc_deg


def getData(plateid, ifudesign, component='stellar', ext='vel'):
   '''
      Get MaNGA MPL-5 data
      (07/14/2017) default extension to open is stellar velocity

      (11/27/2017) adapt file for gas kinematic data

      (01/10/2018) Get MaNGA MPL-6 data
         default component is stellar, default extension is velocity
         instrumental correction for sigma included (if ext == 'disp')

   '''
   
   filename = datapath+'SPX-GAU-MILESHC/'+plateid+'/'+ifudesign+'/manga-'+plateid+'-'+ifudesign+'-MAPS-SPX-GAU-MILESHC.fits.gz'
   datafile = fits.open(filename)

   if component == 'stellar': 
      if ext == 'vel':
         data = np.ma.array(datafile['STELLAR_VEL'].data, dtype='float64', mask=datafile['STELLAR_VEL_MASK'].data, fill_value=np.nan)
         ivar = np.ma.array(datafile['STELLAR_VEL_IVAR'].data, dtype='float64', mask=datafile['STELLAR_VEL_MASK'].data, fill_value=0.)
      elif ext == 'disp':
         sigma_meas = np.array(datafile['STELLAR_SIGMA'].data, 'float64')
         sigma_inst = np.array(datafile['STELLAR_SIGMACORR'].data, 'float64')
         sigma_obs  = np.sqrt( sigma_meas**2 - sigma_inst**2 )
         mask       = np.array(datafile['STELLAR_SIGMA_MASK'].data + ( sigma_meas-sigma_inst < 0 ), 'float64')
         data = np.ma.array(sigma_obs, 'float64', mask=mask, fill_value=np.nan)
         ivar = np.ma.array(datafile['STELLAR_SIGMA_IVAR'].data, 'float64', mask=mask, fill_value=0.)

   else:
      # Build a dictionary with the emission line names to ease selection
      emline = {}
      for k, v in datafile['EMLINE_GFLUX'].header.items():
         if k[0] == 'C':
            try: i = int(k[1:])-1
            except ValueError: continue
            emline[v] = i

      gas_component = component
      if ext == 'vel':
         data = np.ma.array(datafile['EMLINE_GVEL'].data[emline[gas_component]], 'float64', mask=datafile['EMLINE_GVEL_MASK'].data[emline[gas_component]], fill_value=np.nan)
         ivar = np.ma.array(datafile['EMLINE_GVEL_IVAR'].data[emline[gas_component]], 'float64', mask=datafile['EMLINE_GVEL_MASK'].data[emline[gas_component]], fill_value=0.)
      elif ext == 'disp':
         sigma_meas = np.array(datafile['EMLINE_GSIGMA'].data[emline[gas_component]], 'float64')
         sigma_inst = np.array(datafile['EMLINE_INSTSIGMA'].data[emline[gas_component]], 'float64')
         sigma_obs  = np.sqrt( sigma_meas**2 - sigma_inst**2 )
         mask       = np.array(datafile['EMLINE_GSIGMA_MASK'].data[emline[gas_component]] + ( sigma_meas-sigma_inst < 0 ), 'float64')
         data = np.ma.array(sigma_obs, 'float64', mask=mask, fill_value=np.nan)
         ivar = np.ma.array(datafile['EMLINE_GSIGMA_IVAR'].data[emline[gas_component]], 'float64', mask=mask, fill_value=0.)

   datafile.close()

   ### (03/14/2018) test masking bad data for one highly inclined galaxy
   ### (03/16/2018) do for all, but doesn't apply for most galaxies
   low_ivar_pixels = np.where(ivar<1e-4)  ### i.e., sigma > 100 km/s
   huge_ivar_pixels = np.where(ivar>1e10)
   mod_mask = data.mask
   mod_mask[low_ivar_pixels] = 1
   mod_mask[huge_ivar_pixels] = 1

   if plateid+'-'+ifudesign == '8550-12702':
      if component == 'stellar':
         if ext == 'disp': mod_mask[43,46] = 1
   elif plateid+'-'+ifudesign == '8484-12703':
      if component == 'stellar':
         if   ext == 'vel':  mod_mask[65,52] = 1
         elif ext == 'disp': mod_mask[33,7] = 1
   elif plateid+'-'+ifudesign == '8567-12701':
      if component == 'stellar' and ext == 'disp':
         mod_mask[22,46] = 1
   elif plateid+'-'+ifudesign == '8601-12702':
      if component == 'stellar':
         if ext == 'disp': mod_mask[24,14] = 1
   elif plateid+'-'+ifudesign == '8482-12701':
      if component == 'stellar':
         if   ext == 'vel':  mod_mask[30,20] = 1
         elif ext == 'disp': mod_mask[18,20] = 1
   elif plateid+'-'+ifudesign == '9865-12705':
      if component == 'stellar':
         if   ext == 'vel':  mod_mask[23,38] = 1
         elif ext == 'disp': mod_mask[37,46] = 1
   elif plateid+'-'+ifudesign == '8728-12705':
      if component == 'stellar':
         if   ext == 'vel':  mod_mask[34,22] = 1
         elif ext == 'disp': mod_mask[21,24] = 1

   data = np.ma.array(data.data, mask=mod_mask, fill_value=np.nan)
   ivar = np.ma.array(ivar.data, mask=mod_mask, fill_value=0.)

   return data[1:,1:], ivar[1:,1:]


def initModel(plateid, ifudesign, pa_deg, inc_deg, model_dim, ext='vel', V_sys=None, V_max=None, h_rot=None, sigma=None):

   if   ext == 'vel':  simdisk.mkdisk(pa_deg, inc_deg, ext, model_dim[0], V_sys, V_max, h_rot)
   elif ext == 'disp': simdisk.mkdisk(pa_deg, inc_deg, ext, model_dim[0], sigma_cen=sigma)

   filename = modeldir+'PA=%.1f_i=%.1f_'%(pa_deg,inc_deg)+str(ext)+'disk.fits'
   modelfile = fits.open(filename)
   model = modelfile[0].data
   modelfile.close()

   return model


def deproject(model, pa_deg, inc_deg):
   '''
      Correct for inclination
      (07/17/2017) assume inclination reported in NASA-Sloan Atlas
      (08/16/2017) correct R deprojection to concentric ellipses

   '''
   pos_angle   = pa_deg  *np.pi/180
   inclination = inc_deg *np.pi/180

   model_dim = np.shape(model)
   
   r_ip     = np.zeros(model_dim)
   R_gp     = np.zeros(model_dim)
   phi_ip   = np.zeros(model_dim)
   theta_gp = np.zeros(model_dim)
   p_ip     = np.zeros(model_dim)
   bin_assign = np.zeros(model_dim,dtype=int)

   if 0 <= pos_angle < 1.5*np.pi: alpha = pos_angle + 0.5*np.pi
   else:                          alpha = pos_angle % (0.5*np.pi)
   sin_alpha = np.sin(alpha)
   cos_alpha = np.cos(alpha)

   a = 0.5 * 0.8 * model_dim[0]   ### (07/30/2017) more generous cut-off, from a = 0.5 * 0.75 * dim
   b = a * np.cos(inclination)

   model_cen = [model_dim[0]//2, model_dim[1]//2]

   for y in range(model_dim[0]):
      for x in range(model_dim[1]):

         Y = y -model_cen[0]
         X = x -model_cen[1]

         ### radial coordinate r in image plane
         r = np.sqrt( X**2 +Y**2 )
         
         ### azimuthal angle in image plane
         if (X == 0) and (Y == 0):   
            phi = pos_angle +0.5*np.pi
         else:
            phi = np.arctan2(Y,X)
            if (X <= 0) and (Y >= 0): phi -= 0.5*np.pi
            else:                     phi += 1.5*np.pi
         
         ### azimuthal angle in galaxy disk plane
         theta = np.arctan( np.tan(phi-pos_angle+0.5*np.pi) *np.cos(inclination) )
         if phi-pos_angle == 0:
            theta -= 0.5*np.pi
         elif 0 < pos_angle <= np.pi:
            if 0 < phi-pos_angle <= np.pi:   theta += 0.5*np.pi
            else:                            theta += 1.5*np.pi
         elif np.pi < pos_angle < 2*np.pi:
            if pos_angle <= phi <= 2*np.pi:  theta += 0.5*np.pi
            elif 0 <= phi < pos_angle-np.pi: theta += 0.5*np.pi
            else:                            theta += 1.5*np.pi
         
         r_ip[y,x]     = r
         phi_ip[y,x]   = phi
         theta_gp[y,x] = theta

         ### (square of) radial coordinate in galaxy plane (de-projected ellipse) normalized to disk radius R
         p = p_ip[y,x] = (X*cos_alpha +Y*sin_alpha)**2 /a**2 + (X*sin_alpha -Y*cos_alpha)**2 /b**2
         
         ### radial coordinate in galaxy plane
         R = R_gp[y,x] = a * p**0.5
         
         bin_assign[y,x] = assignBin(R)
         
   fits.writeto(modeldir+'PA='+str(pa_deg)+'_i='+str(inc_deg)+'_ellipse.fits',p_ip**0.5,overwrite=True)
   fits.writeto(modeldir+'PA='+str(pa_deg)+'_i='+str(inc_deg)+'_binning.fits',bin_assign,overwrite=True)
   fits.writeto(modeldir+'PA='+str(pa_deg)+'_i='+str(inc_deg)+'_R_gp_mcmc_deproject.fits',R_gp,overwrite=True)
   fits.writeto(modeldir+'PA='+str(pa_deg)+'_i='+str(inc_deg)+'_theta_mcmc_deproject.fits',theta_gp,overwrite=True)
   fits.writeto(modeldir+'PA='+str(pa_deg)+'_i='+str(inc_deg)+'_cos_theta_mcmc_deproject.fits',np.cos(theta_gp),overwrite=True)

   return r_ip, R_gp, phi_ip, theta_gp, p_ip, bin_assign


def assignBin(R):
   for b in range(num_bins):
      if bin_edges[b][0] <= R < bin_edges[b][1]:
         return b
   return -1


def binning(plateid, ifudesign, psf_hwhm):
   
   if   ifudesign[:3] == '127': dim = 74
   elif ifudesign[:2] == '91':  dim = 62
   elif ifudesign[:2] == '61':  dim = 52
   elif ifudesign[:2] == '37':  dim = 44
   elif ifudesign[:2] == '19':  dim = 34
   
   pa_deg, inc_deg = getParams(plateid, ifudesign)
   
   #bin_width = psf_hwhm
   bin_width = round( psf_hwhm * (np.cos(inc_deg*np.pi/180))**-0.5 , 2)
   
   max_R = 1.4 * dim/2
   num_bins = int(max_R //bin_width) +1                     ### (08/18/2017) fit center
   max_R = (num_bins-0.5) * bin_width
   radial_bins = np.linspace(-0.5*bin_width, max_R, num_bins+1)
   
   bin_centers, bin_edges = [], []
   for b in range(num_bins):
      bin_centers.append(round(0.5*(radial_bins[b]+radial_bins[b+1]),2))
      bin_edges.append([round(radial_bins[b],2),round(radial_bins[b+1],2)])
   
   return num_bins, bin_centers, bin_edges


def getFlux(plateid, ifudesign, component='stellar', ext='vel'):

    if component == 'stellar':
        filename = psfpath+plateid+'/stack/manga-'+plateid+'-'+ifudesign+'-LOGCUBE.fits.gz'
        logcube = fits.open(filename)

        # use g-band image
        flux = np.ma.array(logcube['GIMG'].data, mask=None, dtype='float64')
        logcube.close()

        filename = datapath+'SPX-GAU-MILESHC/'+plateid+'/'+ifudesign+'/manga-'+plateid+'-'+ifudesign+'-MAPS-SPX-GAU-MILESHC.fits.gz'
        datafile = fits.open(filename)

        if plateid+'-'+ifudesign == '8135-12701':
            temp = datafile['STELLAR_VEL_MASK'].data
            temp[40:55,35:50] = 0
            flux = np.ma.array(flux.clip(min=0), mask=temp, dtype='float64', fill_value=np.nan)
        else:
            flux = np.ma.array(flux.clip(min=0), mask=datafile['STELLAR_VEL_MASK'].data, dtype='float64', fill_value=np.nan)

    else:
        filename = datapath+'SPX-GAU-MILESHC/'+plateid+'/'+ifudesign+'/manga-'+plateid+'-'+ifudesign+'-MAPS-SPX-GAU-MILESHC.fits.gz'
        datafile = fits.open(filename)

        # Build a dictionary with the emission line names to ease selection
        emline = {}
        for k, v in datafile['EMLINE_GFLUX'].header.items():
            if k[0] == 'C':
                try: i = int(k[1:])-1
                except ValueError: continue
                emline[v] = i

        gas_component = component
        if ext == 'disp':
            flux = np.ma.array(datafile['EMLINE_SFLUX'].data[emline[gas_component]], mask=datafile['EMLINE_SFLUX_MASK'].data[emline[gas_component]], dtype='float64', fill_value=np.nan)   
        elif ext == 'vel':
            dim = np.shape(datafile['EMLINE_SFLUX'].data[emline[gas_component]])
            summed_flux = np.zeros(dim)
            summed_mask = np.zeros(dim)
            for thisline in ['Ha-6564', 'OII-3727', 'OIII-4960', 'OIII-5008']:
                print(np.shape(datafile['EMLINE_SFLUX'].data[emline[thisline]]), np.sum(datafile['EMLINE_SFLUX'].data[emline[thisline]]), np.shape(datafile['EMLINE_SFLUX_MASK'].data[emline[thisline]]), np.sum(datafile['EMLINE_SFLUX_MASK'].data[emline[thisline]].clip(min=0,max=1)))
                summed_flux += datafile['EMLINE_SFLUX'].data[emline[thisline]]
                summed_mask += datafile['EMLINE_SFLUX_MASK'].data[emline[thisline]].clip(min=0,max=1)
            flux = np.ma.array(summed_flux, mask=summed_mask, dtype='float64', fill_value=np.nan)

        datafile.close()

    flux = flux[1:,1:]
    gaussian_kernel = np.array(Gaussian2DKernel(2.35))
    test = convolve_fft(flux, gaussian_kernel, nan_treatment='interpolate')
    flux = np.ma.array(test*(flux.mask)+flux.data*(1-flux.mask), mask=flux.mask)

    return flux


def getPSF(plateid, ifudesign, component='stellar', ext='vel'):

    filename = psfpath+plateid+'/stack/manga-'+plateid+'-'+ifudesign+'-LOGCUBE.fits.gz'
    psf_file = fits.open(filename)

    # use g-band psf
    psf = np.ma.array(psf_file['GPSF'].data, dtype='float64')
    psf_file.close()

    return psf[1:,1:]


def getGFWHM(plateid, ifudesign):
    
    filename = datapath+'SPX-GAU-MILESHC/'+plateid+'/'+ifudesign+'/manga-'+plateid+'-'+ifudesign+'-MAPS-SPX-GAU-MILESHC.fits.gz'
    datafile = fits.open(filename)

    gfwhm = float(datafile[0].header['GFWHM'])
    datafile.close()

    return gfwhm


def convolveFFT(intr_model, psf, data_mask):

    output = signal.convolve(psf, intr_model, mode='same', method='fft')
    return np.ma.array(output, mask=data_mask, dtype='float64')


def computeChiSquared(data, ivar, model):

   if intr_var == 0:
      masked_ivar = np.ma.filled(ivar)
      dim = np.count_nonzero(masked_ivar)
      chi2 = np.sum( masked_ivar * (data-model)**2 )

   else:
      ### (09/27/2017; KG) add intrinsic variance to denominator in chi-squared calculation
      new_ivar = np.ma.filled( (ivar**-1 + intr_var)**-1, 0. )
      dim = np.count_nonzero(new_ivar)
      chi2 = np.sum( new_ivar * (data-model)**2 )

   return chi2, dim


def vary_V_t(model, target_bin, sigma_V_t):

   model_dim = np.shape(model[0][0])
   model_cen = [model_dim[0]//2, model_dim[1]//2]
   
   disk_param = model[6]
   R_gp, theta_gp, bin_assign = disk_param[0], disk_param[1], disk_param[3]
   
   V_sys = model[2]
   phi_bar = model[3]
   inclination = model[4]
   
   vel_model  = model[0][0] -V_sys
   
   previous_value = model[1][target_bin][0]
   proposal_sigma = sigma_V_t
   
   if not target_bin == 0:
      proposed_value = np.random.normal(previous_value, proposal_sigma)

      if target_bin == num_bins-1:                                                  ### right edge
         interp_range      = [bin_centers[target_bin-1], bin_centers[target_bin]]
         interp_V_t_values = [model[1][target_bin-1][0], proposed_value]

      else:
         interp_range      = [bin_centers[target_bin-1], bin_centers[target_bin], bin_centers[target_bin+1]]
         interp_V_t_values = [model[1][target_bin-1][0], proposed_value,          model[1][target_bin+1][0]]
         
      sin_i = np.sin(inclination)
      theta_bar = theta_gp -phi_bar
      cos_2thetaBar = np.cos(2*theta_bar)
      sin_2thetaBar = np.sin(2*theta_bar)
      cos_theta = np.cos(theta_gp)
      sin_theta = np.sin(theta_gp)
      
      V_2t, V_2r = model[1][target_bin][2:4]

      for y in range(model_dim[0]):
         for x in range(model_dim[1]):
         
            R = R_gp[y,x]
            if interp_range[0] <= R <= interp_range[len(interp_range)-1]:     ### (08/21/2017)
               vel_model[y,x] = sin_i * ( \
                                         np.interp(R, interp_range, interp_V_t_values) * cos_theta[y,x] \
                                         - V_2t * cos_2thetaBar[y,x] * cos_theta[y,x] \
                                         - V_2r * sin_2thetaBar[y,x] * sin_theta[y,x] \
                                         )
               
      model_out = [ vel_model +V_sys, model[0][1] ]
      return model_out, proposed_value

   else: return model[0], previous_value


def vary_V(model, target_bin, sigma_V_t, sigma_V_2t, sigma_V_2r):
   
   model_dim = np.shape(model[0][0])
   model_cen = [model_dim[0]//2, model_dim[1]//2]
   
   disk_param = model[6]
   R_gp, theta_gp, bin_assign = disk_param[0], disk_param[1], disk_param[3]
   
   V_sys = model[2]
   phi_bar = model[3]
   inclination = model[4]
   
   vel_model  = model[0][0] -V_sys
   
   previous_V_t                 = model[1][target_bin][0]
   previous_V_2t, previous_V_2r = model[1][target_bin][2:4]
   
   if not target_bin == 0:
      proposed_V_t  = np.random.normal(previous_V_t,  sigma_V_t)
      proposed_V_2t = np.random.normal(previous_V_2t, sigma_V_2t)
      proposed_V_2r = np.random.normal(previous_V_2r, sigma_V_2r)
      
      if target_bin == num_bins-1:                                                  ### right edge
         interp_range       = [bin_centers[target_bin-1], bin_centers[target_bin]]
         interp_V_t_values  = [model[1][target_bin-1][0], proposed_V_t]
         interp_V_2t_values = [model[1][target_bin-1][2], proposed_V_2t]
         interp_V_2r_values = [model[1][target_bin-1][3], proposed_V_2r]
      
      else:
         interp_range       = [bin_centers[target_bin-1], bin_centers[target_bin], bin_centers[target_bin+1]]
         interp_V_t_values  = [model[1][target_bin-1][0], proposed_V_t,            model[1][target_bin+1][0]]
         interp_V_2t_values = [model[1][target_bin-1][2], proposed_V_2t,           model[1][target_bin+1][2]]
         interp_V_2r_values = [model[1][target_bin-1][3], proposed_V_2r,           model[1][target_bin+1][3]]
      
      sin_i = np.sin(inclination)
      theta_bar = theta_gp -phi_bar
      cos_2thetaBar = np.cos(2*theta_bar)
      sin_2thetaBar = np.sin(2*theta_bar)
      cos_theta = np.cos(theta_gp)
      sin_theta = np.sin(theta_gp)

      for y in range(model_dim[0]):
         for x in range(model_dim[1]):
            
            R = R_gp[y,x]
            if interp_range[0] <= R <= interp_range[len(interp_range)-1]:     ### (08/21/2017)
               vel_model[y,x] = sin_i * \
                   ( np.interp(R, interp_range, interp_V_t_values)  * cos_theta[y,x] \
                   - np.interp(R, interp_range, interp_V_2t_values) * cos_2thetaBar[y,x] * cos_theta[y,x] \
                   - np.interp(R, interp_range, interp_V_2r_values) * sin_2thetaBar[y,x] * sin_theta[y,x] \
                   )
      
      model_out = [ vel_model +V_sys, model[0][1] ]
      return model_out, proposed_V_t, proposed_V_2t, proposed_V_2r
   
   else: return model[0], previous_V_t, previous_V_2t, previous_V_2r


def vary_sigma_V(model, target_bin, sigma_disp):
   
   model_dim = np.shape(model[0][0])
   model_cen = [model_dim[0]//2, model_dim[1]//2]
   
   disk_param = model[6]
   R_gp, theta_gp, bin_assign = disk_param[0], disk_param[1], disk_param[3]
   
   inclination = model[4]
   
   disp_model = model[0][1]
   
   previous_value = model[1][target_bin][1]
   proposal_sigma = sigma_disp
   
   proposed_value = np.random.normal(previous_value, proposal_sigma)

   if target_bin == 0:                                                              ### left edge
      interp_range          = [bin_centers[target_bin], bin_centers[target_bin+1]]
      interp_sigma_V_values = [proposed_value,          model[1][target_bin+1][1]]
   
   elif target_bin == num_bins-1:                                                   ### right edge
      interp_range          = [bin_centers[target_bin-1], bin_centers[target_bin]]
      interp_sigma_V_values = [model[1][target_bin-1][1], proposed_value]
   
   else:
      interp_range          = [bin_centers[target_bin-1], bin_centers[target_bin], bin_centers[target_bin+1]]
      interp_sigma_V_values = [model[1][target_bin-1][1], proposed_value,          model[1][target_bin+1][1]]

   for y in range(model_dim[0]):
      for x in range(model_dim[1]):
         
         R = R_gp[y,x]
         if interp_range[0] <= R <= interp_range[len(interp_range)-1]:     ### (08/21/2017)

            disp_model[y,x] = np.interp(R, interp_range, interp_sigma_V_values)

   model_out = [ model[0][0], disp_model ]
   return model_out, proposed_value


def vary_V_sys(model, sigma_Vsys):

   previous_Vsys = model[2]
   
   ### gaussian proposal distribution
   proposed_Vsys = np.random.normal(previous_Vsys, sigma_Vsys)
   
   model_out = model[0][0] -previous_Vsys +proposed_Vsys

   return [model_out, model[0][1]], proposed_Vsys


def vary_phi_bar(model, sigma_phiBar):

   previous_phiBar = model[3]

   ### gaussian proposal distribution
   proposed_phiBar = np.random.normal(previous_phiBar, sigma_phiBar)
   
   disk_param = model[6]
   R_gp, theta_gp, bin_assign = disk_param[0], disk_param[1], disk_param[3]

   inclination = model[4]
   sin_i = np.sin(inclination)

   theta_bar = theta_gp -proposed_phiBar
   cos_2thetaBar = np.cos(2*theta_bar)
   sin_2thetaBar = np.sin(2*theta_bar)
   cos_theta = np.cos(theta_gp)
   sin_theta = np.sin(theta_gp)
   
   V_sys = model[2]
   
   model_dim = np.shape(model[0][0])
   model_out = np.zeros(model_dim)
   
   for y in range(model_dim[0]):
      for x in range(model_dim[1]):
         
         R         = R_gp[y,x]
         which_bin = bin_assign[y,x]
         
         if which_bin == 0:                                                          ### left edge
            interp_range       = [bin_centers[which_bin], bin_centers[which_bin+1]]
            interp_V_t_values  = [model[1][which_bin][0], model[1][which_bin+1][0]]
            interp_V_2t_values = [model[1][which_bin][2], model[1][which_bin+1][2]]
            interp_V_2r_values = [model[1][which_bin][3], model[1][which_bin+1][3]]
         
         elif which_bin == num_bins-1:                                              ### right edge
            interp_range       = [bin_centers[which_bin-1], bin_centers[which_bin]]
            interp_V_t_values  = [model[1][which_bin-1][0], model[1][which_bin][0]]
            interp_V_2t_values = [model[1][which_bin-1][2], model[1][which_bin][2]]
            interp_V_2r_values = [model[1][which_bin-1][3], model[1][which_bin][3]]

         else:
            interp_range       = [bin_centers[which_bin-1], bin_centers[which_bin], bin_centers[which_bin+1]]
            interp_V_t_values  = [model[1][which_bin-1][0], model[1][which_bin][0], model[1][which_bin+1][0]]
            interp_V_2t_values = [model[1][which_bin-1][2], model[1][which_bin][2], model[1][which_bin+1][2]]
            interp_V_2r_values = [model[1][which_bin-1][3], model[1][which_bin][3], model[1][which_bin+1][3]]

         model_out[y,x] = sin_i * ( \
                            np.interp(R, interp_range, interp_V_t_values)  * cos_theta[y,x] \
                          - np.interp(R, interp_range, interp_V_2t_values) * cos_2thetaBar[y,x] * cos_theta[y,x] \
                          - np.interp(R, interp_range, interp_V_2r_values) * sin_2thetaBar[y,x] * sin_theta[y,x] \
                        )
   
   return [model_out +V_sys, model[0][1]], proposed_phiBar


def vary_inclination(model, sigma_inc):

   model_dim = np.shape(model[0][0])
   model_cen = [model_dim[0]//2, model_dim[1]//2]

   previous_inc = model[4]
   
   ### gaussian proposal distribution
   proposed_inc = np.random.normal(previous_inc, sigma_inc)
   sin_i = np.sin(proposed_inc)
   cos_i = np.cos(proposed_inc)
   
   disk_param = model[6]
   pos_angle, phi_ip = disk_param[4], disk_param[7]
   
   proposed_theta_gp = np.zeros(model_dim)
   proposed_R_gp     = np.zeros(model_dim)
   proposed_bins     = np.zeros(model_dim,dtype=int)

   if 0 <= pos_angle < 1.5*np.pi: alpha = pos_angle + 0.5*np.pi
   else:                          alpha = pos_angle % (0.5*np.pi)
   sin_alpha = np.sin(alpha)
   cos_alpha = np.cos(alpha)

   a = 0.5 * 0.8 * model_dim[0]   ### (07/30/2017) more generous cut-off, from a = 0.5 * 0.75 * dim
   b = a * cos_i

   for y in range(model_dim[0]):
      for x in range(model_dim[1]):

         Y = y -model_cen[0]
         X = x -model_cen[1]

         ### azimuthal angle phi in image plane
         phi = phi_ip[y,x]

         ### azimuthal angle in galaxy disk plane
         theta = np.arctan( np.tan(phi-pos_angle+0.5*np.pi) *cos_i )
         if phi-pos_angle == 0:
            theta -= 0.5*np.pi
         elif 0 < pos_angle <= np.pi:
            if 0 < phi-pos_angle <= np.pi:   theta += 0.5*np.pi
            else:                            theta += 1.5*np.pi
         elif np.pi < pos_angle < 2*np.pi:
            if pos_angle <= phi <= 2*np.pi:  theta += 0.5*np.pi
            elif 0 <= phi < pos_angle-np.pi: theta += 0.5*np.pi
            else:                            theta += 1.5*np.pi

         proposed_theta_gp[y,x] = theta

         ### (square of) radial coordinate in galaxy plane (de-projected ellipse) normalized to disk radius R
         p = (X*cos_alpha +Y*sin_alpha)**2 /a**2 + (X*sin_alpha -Y*cos_alpha)**2 /b**2
         
         ### radial coordinate in galaxy plane
         R = proposed_R_gp[y,x] = a * p**0.5
         
         proposed_bins[y,x] = assignBin(R)

   V_sys, phi_bar = model[2], model[3]
   
   theta_bar = proposed_theta_gp -phi_bar            ### phi_bar is defined in galaxy plane, following Spekkens & Sellwood (2007)
   cos_2thetaBar = np.cos(2*theta_bar)
   sin_2thetaBar = np.sin(2*theta_bar)
   cos_theta = np.cos(proposed_theta_gp)
   sin_theta = np.sin(proposed_theta_gp)

   model_out = np.zeros(model_dim)
   
   for y in range(model_dim[0]):
      for x in range(model_dim[1]):
         
         R         = proposed_R_gp[y,x]
         which_bin = proposed_bins[y,x]
         
         if which_bin == 0:                                                          ### left edge
            interp_range       = [bin_centers[which_bin], bin_centers[which_bin+1]]
            interp_V_t_values  = [model[1][which_bin][0], model[1][which_bin+1][0]]
            interp_V_2t_values = [model[1][which_bin][2], model[1][which_bin+1][2]]
            interp_V_2r_values = [model[1][which_bin][3], model[1][which_bin+1][3]]
         
         elif which_bin == num_bins-1:                                               ### right edge
            interp_range       = [bin_centers[which_bin-1], bin_centers[which_bin]]
            interp_V_t_values  = [model[1][which_bin-1][0], model[1][which_bin][0]]
            interp_V_2t_values = [model[1][which_bin-1][2], model[1][which_bin][2]]
            interp_V_2r_values = [model[1][which_bin-1][3], model[1][which_bin][3]]
         
         else:
            interp_range       = [bin_centers[which_bin-1], bin_centers[which_bin], bin_centers[which_bin+1]]
            interp_V_t_values  = [model[1][which_bin-1][0], model[1][which_bin][0], model[1][which_bin+1][0]]
            interp_V_2t_values = [model[1][which_bin-1][2], model[1][which_bin][2], model[1][which_bin+1][2]]
            interp_V_2r_values = [model[1][which_bin-1][3], model[1][which_bin][3], model[1][which_bin+1][3]]
      
         model_out[y,x] = sin_i * ( \
                                   np.interp(R, interp_range, interp_V_t_values)  * cos_theta[y,x] \
                                   - np.interp(R, interp_range, interp_V_2t_values) * cos_2thetaBar[y,x] * cos_theta[y,x] \
                                   - np.interp(R, interp_range, interp_V_2r_values) * sin_2thetaBar[y,x] * sin_theta[y,x] \
                                   )

   return [model_out +V_sys, model[0][1]], proposed_inc, [proposed_R_gp, proposed_theta_gp, proposed_bins]


### (10/03/2017) start fitting disk position angle
def vary_PA(model, sigma_PA):

   model_dim = np.shape(model[0][0])
   model_cen = [model_dim[0]//2, model_dim[1]//2]

   previous_PA = model[5]

   ### gaussian proposal distribution
   proposed_PA = np.random.normal(previous_PA, sigma_PA)
   proposed_PA %= (2*np.pi)

   if 0 <= proposed_PA < 1.5*np.pi: alpha = proposed_PA + 0.5*np.pi
   else:                            alpha = proposed_PA % (0.5*np.pi)
   sin_alpha = np.sin(alpha)
   cos_alpha = np.cos(alpha)

   inclination = model[4]
   sin_i = np.sin(inclination)
   cos_i = np.cos(inclination)
   
   a = 0.5 * 0.8 * model_dim[0]   ### (07/30/2017) more generous cut-off, from a = 0.5 * 0.75 * dim
   b = a * cos_i
   
   disk_param = model[6]
   phi_ip = disk_param[7]
   
   proposed_theta_gp = np.zeros(model_dim)
   proposed_R_gp     = np.zeros(model_dim)
   proposed_bins     = np.zeros(model_dim,dtype=int)

   for y in range(model_dim[0]):
      for x in range(model_dim[1]):
      
         Y = y -model_cen[0]
         X = x -model_cen[1]
         
         ### azimuthal angle phi in image plane
         phi = phi_ip[y,x]

         ### azimuthal angle in galaxy disk plane
         theta = np.arctan( np.tan(phi-proposed_PA+0.5*np.pi) *cos_i )
         if phi-proposed_PA == 0:
            theta -= 0.5*np.pi
         elif 0 < proposed_PA <= np.pi:
            if 0 < phi-proposed_PA <= np.pi:   theta += 0.5*np.pi
            else:                              theta += 1.5*np.pi
         elif np.pi < proposed_PA < 2*np.pi:
            if proposed_PA <= phi <= 2*np.pi:  theta += 0.5*np.pi
            elif 0 <= phi < proposed_PA-np.pi: theta += 0.5*np.pi
            else:                              theta += 1.5*np.pi

         proposed_theta_gp[y,x] = theta

         ### (square of) radial coordinate in galaxy plane (de-projected ellipse) normalized to disk radius R
         p = (X*cos_alpha +Y*sin_alpha)**2 /a**2 + (X*sin_alpha -Y*cos_alpha)**2 /b**2

         ### radial coordinate in galaxy plane
         R = proposed_R_gp[y,x] = a * p**0.5

         proposed_bins[y,x] = assignBin(R)

   V_sys, phi_bar = model[2], model[3]
   
   theta_bar = proposed_theta_gp -phi_bar            ### phi_bar is defined in galaxy plane, following Spekkens & Sellwood (2007)
   cos_2thetaBar = np.cos(2*theta_bar)
   sin_2thetaBar = np.sin(2*theta_bar)
   cos_theta = np.cos(proposed_theta_gp)
   sin_theta = np.sin(proposed_theta_gp)
      
   V_model_out     = np.zeros(model_dim)
   sigma_model_out = np.zeros(model_dim)
   
   for y in range(model_dim[0]):
      for x in range(model_dim[1]):
         
         R         = proposed_R_gp[y,x]
         which_bin = proposed_bins[y,x]
         
         if which_bin == 0:                                                             ### left edge
            interp_range          = [bin_centers[which_bin], bin_centers[which_bin+1]]
            interp_V_t_values     = [model[1][which_bin][0], model[1][which_bin+1][0]]
            interp_sigma_V_values = [model[1][which_bin][1], model[1][which_bin+1][1]]
            interp_V_2t_values    = [model[1][which_bin][2], model[1][which_bin+1][2]]
            interp_V_2r_values    = [model[1][which_bin][3], model[1][which_bin+1][3]]
         
         elif which_bin == num_bins-1:                                                  ### right edge
            interp_range          = [bin_centers[which_bin-1], bin_centers[which_bin]]
            interp_V_t_values     = [model[1][which_bin-1][0], model[1][which_bin][0]]
            interp_sigma_V_values = [model[1][which_bin-1][1], model[1][which_bin][1]]
            interp_V_2t_values    = [model[1][which_bin-1][2], model[1][which_bin][2]]
            interp_V_2r_values    = [model[1][which_bin-1][3], model[1][which_bin][3]]
         
         else:
            interp_range          = [bin_centers[which_bin-1], bin_centers[which_bin], bin_centers[which_bin+1]]
            interp_V_t_values     = [model[1][which_bin-1][0], model[1][which_bin][0], model[1][which_bin+1][0]]
            interp_sigma_V_values = [model[1][which_bin-1][1], model[1][which_bin][1], model[1][which_bin+1][1]]
            interp_V_2t_values    = [model[1][which_bin-1][2], model[1][which_bin][2], model[1][which_bin+1][2]]
            interp_V_2r_values    = [model[1][which_bin-1][3], model[1][which_bin][3], model[1][which_bin+1][3]]
         
         V_model_out[y,x] = sin_i * ( \
                                   np.interp(R, interp_range, interp_V_t_values)  * cos_theta[y,x] \
                                   - np.interp(R, interp_range, interp_V_2t_values) * cos_2thetaBar[y,x] * cos_theta[y,x] \
                                   - np.interp(R, interp_range, interp_V_2r_values) * sin_2thetaBar[y,x] * sin_theta[y,x] \
                                   )
         sigma_model_out[y,x] = np.interp(R, interp_range, interp_sigma_V_values)

   return [V_model_out +V_sys, sigma_model_out], proposed_PA, [proposed_R_gp, proposed_theta_gp, proposed_bins]


def proposeSwaps(plateid, ifudesign, component, data, ivar, num_chains, temp_ladder, last_models, psf, flux, conv_norm, this_iter):
   print('      ######  proposeSwaps() reached at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   next_models = np.copy(last_models)

   record_swaps = open(testdir+plateid+'-'+ifudesign+'/'+component+'/swaps/manga_'+plateid+'_'+ifudesign+'_swaps_it%04d.dat'%(this_iter+1),'w')
   record_swaps.write('#  '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())+' \n')
   record_swaps.write('#\n')
   record_swaps.write('#    MaNGA plate ID:   '+plateid+' \n')
   record_swaps.write('#    MaNGA IFU design: '+ifudesign+' \n')
   record_swaps.write('#\n')
   record_swaps.write('#    iteration %.0f \n'%(this_iter+1))
      
   for k in range(num_chains-1):
      record_swaps.write('#\n')
   
      ### (08/16/2017) ladder climb-down
      swap_elements = [num_chains-k-2, num_chains-k-1]
      beta_i = temp_ladder[swap_elements[0]] **-1
      beta_j = temp_ladder[swap_elements[1]] **-1
      
      intr_vel_model_i = last_models[swap_elements[0]][0][0]
      intr_vel_model_j = last_models[swap_elements[1]][0][0]
      intr_disp_model_i = last_models[swap_elements[0]][0][1]
      intr_disp_model_j = last_models[swap_elements[1]][0][1]

      vel_avg_i = convolveFFT(intr_vel_model_i, psf/np.sum(psf), None)
      vel_avg_j = convolveFFT(intr_vel_model_j, psf/np.sum(psf), None)

      if component == 'stellar':
         conv_vel_model_i = convolveFFT(intr_vel_model_i*flux.data, psf, data[0].mask) * conv_norm
         conv_vel_model_j = convolveFFT(intr_vel_model_j*flux.data, psf, data[0].mask) * conv_norm
         conv_disp_model_i = np.sqrt( convolveFFT((intr_disp_model_i**2+(intr_vel_model_i-vel_avg_i)**2)*flux.data, psf, data[1].mask) * conv_norm )
         conv_disp_model_j = np.sqrt( convolveFFT((intr_disp_model_j**2+(intr_vel_model_j-vel_avg_j)**2)*flux.data, psf, data[1].mask) * conv_norm )
      else:
         conv_vel_model_i = convolveFFT(intr_vel_model_i*flux[0].data, psf, data[0].mask) * conv_norm[0]
         conv_vel_model_j = convolveFFT(intr_vel_model_j*flux[0].data, psf, data[0].mask) * conv_norm[0]
         conv_disp_model_i = np.sqrt( convolveFFT((intr_disp_model_i**2+(intr_vel_model_i-vel_avg_i)**2)*flux[1].data, psf, data[1].mask) * conv_norm[1] )
         conv_disp_model_j = np.sqrt( convolveFFT((intr_disp_model_j**2+(intr_vel_model_j-vel_avg_j)**2)*flux[1].data, psf, data[1].mask) * conv_norm[1] )

      ### (08/18/2017) use data.mask instead of truncating to ellipse
      vel_chi2_i, vel_dim = computeChiSquared(data[0], ivar[0], np.ma.filled(conv_vel_model_i,np.nan))
      vel_chi2_j, vel_dim = computeChiSquared(data[0], ivar[0], np.ma.filled(conv_vel_model_j,np.nan))
      
      disp_chi2_i, disp_dim = computeChiSquared(data[1], ivar[1], np.ma.filled(conv_disp_model_i,np.nan))
      disp_chi2_j, disp_dim = computeChiSquared(data[1], ivar[1], np.ma.filled(conv_disp_model_j,np.nan))
      
      chi2_i = vel_chi2_i + disp_chi2_i
      chi2_j = vel_chi2_j + disp_chi2_j

      ### (09/18/2017) add smoothing term to chi-squared
      smooth_i = smoothing(bin_centers, last_models[swap_elements[0]][1], component)
      smooth_j = smoothing(bin_centers, last_models[swap_elements[1]][1], component)
      smoothing_term_i = sum(smooth_i[0]) +smooth_i[1]
      smoothing_term_j = sum(smooth_j[0]) +smooth_j[1]

      arg = (-chi2_i -smoothing_term_i +chi2_j +smoothing_term_j)       ### (09/18/2017) add smoothing term to chi-squared
      swap_prob = np.exp(arg *(beta_j -beta_i))

      if swap_prob > np.random.rand():    # accept swap
         next_models[swap_elements[1]] = np.copy(last_models[swap_elements[0]])
         next_models[swap_elements[0]] = np.copy(last_models[swap_elements[1]])
         last_models[swap_elements[0]] = np.copy(next_models[swap_elements[0]])
         print('   ###### ('+plateid+'-'+ifudesign+', '+component+')  proposed swap between log T = {%.2f, %.2f} ACCEPTED'%(np.log10(temp_ladder[swap_elements[0]]),np.log10(temp_ladder[swap_elements[1]])))
         record_swaps.write('   proposed swap between log T = {%.2f, %.2f} ACCEPTED \n'%(np.log10(temp_ladder[swap_elements[0]]),np.log10(temp_ladder[swap_elements[1]])))
   
      else:
         print('   ###### ('+plateid+'-'+ifudesign+', '+component+')  proposed swap between log T = {%.2f, %.2f} REJECTED'%(np.log10(temp_ladder[swap_elements[0]]),np.log10(temp_ladder[swap_elements[1]])))
         record_swaps.write('   proposed swap between log T = {%.2f, %.2f} REJECTED \n'%(np.log10(temp_ladder[swap_elements[0]]),np.log10(temp_ladder[swap_elements[1]])))
      
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')  swap acceptance probability: %.2e' % min(swap_prob,1))

      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')    TOTAL: ')
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')       chi2_{i} = %.3f, chi2_{j} = %.3f' % (chi2_i, chi2_j))
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')       reduced chi2_{i} = %.8f, reduced chi2_{j} = %.8f' % (chi2_i/(vel_dim+disp_dim-(n_params_vel+n_params_disp+num_global_params)), chi2_j/(vel_dim+disp_dim-(n_params_vel+n_params_disp+num_global_params))))
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')       with smoothing term: ')
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')          chi2_{i} = %.3f, chi2_{j} = %.3f' % (chi2_i+smoothing_term_i, chi2_j+smoothing_term_j))
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')          reduced chi2_{i} = %.8f, reduced chi2_{j} = %.8f' % ((chi2_i+smoothing_term_i)/(vel_dim+disp_dim-(n_params_vel+n_params_disp+num_global_params)), (chi2_j+smoothing_term_j)/(vel_dim+disp_dim-(n_params_vel+n_params_disp+num_global_params))))

      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')    VELOCITY: ')
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')       chi2_{i} = %.3f, chi2_{j} = %.3f' % (vel_chi2_i, vel_chi2_j))
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')       reduced chi2_{i} = %.8f, reduced chi2_{j} = %.8f' % (vel_chi2_i/(vel_dim-(n_params_vel+num_global_params)), vel_chi2_j/(vel_dim-(n_params_vel+num_global_params))))
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')       with smoothing term: ')
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')          chi2_{i} = %.3f, chi2_{j} = %.3f' % (vel_chi2_i+sum(smooth_i[0]), vel_chi2_j+sum(smooth_j[0])))
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')          reduced chi2_{i} = %.8f, reduced chi2_{j} = %.8f' % ((vel_chi2_i+sum(smooth_i[0]))/(vel_dim-(n_params_vel+num_global_params)), (vel_chi2_j+sum(smooth_j[0]))/(vel_dim-(n_params_vel+num_global_params))))
      
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')    DISPERSION: ')
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')       chi2_{i} = %.3f, chi2_{j} = %.3f' % (disp_chi2_i, disp_chi2_j))
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')       reduced chi2_{i} = %.8f, reduced chi2_{j} = %.8f' % (disp_chi2_i/(disp_dim-(n_params_disp+num_global_params)), disp_chi2_j/(disp_dim-(n_params_disp+num_global_params))))
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')       with smoothing term: ')
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')          chi2_{i} = %.3f, chi2_{j} = %.3f' % (disp_chi2_i+smooth_i[1], disp_chi2_j+smooth_j[1]))
      print('   ###### ('+plateid+'-'+ifudesign+', '+component+')          reduced chi2_{i} = %.8f, reduced chi2_{j} = %.8f' % ((disp_chi2_i+smooth_i[1])/(disp_dim-(n_params_disp+num_global_params)), (disp_chi2_j+smooth_j[1])/(disp_dim-(n_params_disp+num_global_params))))
      
      print('')
      
      record_swaps.write('   swap acceptance probability: %.2e \n' % min(swap_prob,1))

      record_swaps.write('    TOTAL: ')
      record_swaps.write('                      chi2_{i} = %.2f, chi2_{j} = %.2f ' % (chi2_i, chi2_j))
      record_swaps.write('      reduced chi2_{i} = %.4f, reduced chi2_{j} = %.4f \n' % (chi2_i/(vel_dim+disp_dim-(n_params_vel+n_params_disp+num_global_params)), chi2_j/(vel_dim+disp_dim-(n_params_vel+n_params_disp+num_global_params))))
      record_swaps.write('      with smoothing term: ')
      record_swaps.write('      chi2_{i} = %.2f, chi2_{j} = %.2f ' % (chi2_i+smoothing_term_i, chi2_j+smoothing_term_j))
      record_swaps.write('      reduced chi2_{i} = %.4f, reduced chi2_{j} = %.4f \n' % ((chi2_i+smoothing_term_i)/(vel_dim+disp_dim-(n_params_vel+n_params_disp+num_global_params)), (chi2_j+smoothing_term_j)/(vel_dim+disp_dim-(n_params_vel+n_params_disp+num_global_params))))

      record_swaps.write('    VELOCITY: ')
      record_swaps.write('                   chi2_{i} = %.2f, chi2_{j} = %.2f ' % (vel_chi2_i, vel_chi2_j))
      record_swaps.write('      reduced chi2_{i} = %.4f, reduced chi2_{j} = %.4f \n' % (vel_chi2_i/(vel_dim-(n_params_vel+num_global_params)), vel_chi2_j/(vel_dim-(n_params_vel+num_global_params))))
      record_swaps.write('      with smoothing term: ')
      record_swaps.write('      chi2_{i} = %.2f, chi2_{j} = %.2f ' % (vel_chi2_i+sum(smooth_i[0]), vel_chi2_j+sum(smooth_j[0])))
      record_swaps.write('      reduced chi2_{i} = %.4f, reduced chi2_{j} = %.4f \n' % ((vel_chi2_i+sum(smooth_i[0]))/(vel_dim-(n_params_vel+num_global_params)), (vel_chi2_j+sum(smooth_j[0]))/(vel_dim-(n_params_vel+num_global_params))))

      record_swaps.write('    DISPERSION: ')
      record_swaps.write('                 chi2_{i} = %.2f, chi2_{j} = %.2f ' % (disp_chi2_i, disp_chi2_j))
      record_swaps.write('      reduced chi2_{i} = %.4f, reduced chi2_{j} = %.4f \n' % (disp_chi2_i/(disp_dim-(n_params_disp+num_global_params)), disp_chi2_j/(disp_dim-(n_params_disp+num_global_params))))
      record_swaps.write('      with smoothing term: ')
      record_swaps.write('      chi2_{i} = %.2f, chi2_{j} = %.2f ' % (disp_chi2_i+smooth_i[1], disp_chi2_j+smooth_j[1]))
      record_swaps.write('      reduced chi2_{i} = %.4f, reduced chi2_{j} = %.4f \n' % ((disp_chi2_i+smooth_i[1])/(disp_dim-(n_params_disp+num_global_params)), (disp_chi2_j+smooth_j[1])/(disp_dim-(n_params_disp+num_global_params))))

   record_swaps.write('#  \n')
   record_swaps.close()

   ### return list of length=nchain with models post-swap
   return next_models
   

def ptWrapper(plateid, ifudesign, component, temp_ladder, num_chains, burn_in, data, ivar, models, psf, flux, conv_norm, this_iter, verbose=False):

   manager = mp.Manager()
   output = manager.Queue()
   proc = []

   for walker in range(num_chains):
      np.random.seed()
      model = models[walker]
      T = temp_ladder[walker]
      if walker == 0:
         print(sigma_V_t, sigma_disp, sigma_V_2t, sigma_V_2r, sigma_Vsys, round(sigma_phiBar*180/np.pi,2), round(sigma_inc*180/np.pi,2), round(sigma_PA*180/np.pi,2))
      p = mp.Process(target=mcmcFit, args=(plateid, ifudesign, component, data, ivar, model, psf, flux, conv_norm, walker, T, this_iter, output, burn_in, verbose))
      proc.append(p)
      p.start()

   for p in proc: p.join()

   results = [output.get() for p in proc]
   results.sort()

   print('')
   last_models = []
   for wnum in range(num_chains):
      walker_position = results[wnum][1][1]
      Vsys_position = results[wnum][1][2]
      phi_bar_position = round(results[wnum][1][3] *180/np.pi, 4)
      inclination_position = round(results[wnum][1][4] *180/np.pi, 4)
      PA_position = round(results[wnum][1][5] *180/np.pi, 4)
      
      print('('+plateid+'-'+ifudesign+') walker '+str(wnum), 'log T = %.2f'%(np.log10(temp_ladder[wnum])), walker_position, 'V_sys = %.3f km/s'%Vsys_position, 'phi_b = %.3f deg'%phi_bar_position, 'inc = %.3f deg'%inclination_position, 'PA = %.3f deg'%PA_position)
      print('')
      last_models.append(results[wnum][1])

   if swap:       ### (08/04/2017) go back to PT
      new_models = proposeSwaps(plateid, ifudesign, component, data, ivar, num_chains, temp_ladder, last_models, psf, flux, conv_norm, this_iter)
      return new_models

   else:
      return last_models      ### (08/03/2017) test doing away with different chain temperatures, use parallel processes to generate more cold-chain samples instead


def ptRuns(plateid, ifudesign, component, max_iterations, last_iteration=None, verbose=False, fft=True):
   
   pa_deg, inc_deg = getParams(plateid, ifudesign)
   
   if last_iteration is None:
      ### (01/22/2018) switch to using generic initial model full-time
      V_sys, V_max, h_rot, sigma_V_cen = 20., 200., 10., 150.
   
   psf       = getPSF(plateid, ifudesign)
   model_dim = np.shape(psf)

   if component == 'stellar':
      flux = getFlux(plateid, ifudesign, component)
      conv_norm = 1./convolveFFT(flux.data, psf, flux.mask).data
   else:
      flux = ( getFlux(plateid, ifudesign, component, 'vel') , getFlux(plateid, ifudesign, component, 'disp') )
      conv_norm = ( 1./convolveFFT(flux[0].data, psf, flux[0].mask).data , 1./convolveFFT(flux[1].data, psf, flux[1].mask).data )

   vel_data, vel_ivar = getData(plateid, ifudesign, component)

   if last_iteration is None:
      intr_vel_model = initModel(plateid, ifudesign, pa_deg, inc_deg, model_dim, 'vel', V_sys, V_max, h_rot)
      r_ip, R_gp, phi_ip, theta_gp, ellipse, bin_assign = deproject(intr_vel_model, pa_deg, inc_deg)
      a = 0.5 *0.8 *model_dim[0]
      sigma_cen = sigma_V_cen

      ### (07/05/2018) instead of negative inclination implement as adding pi to position angle
      v_a, v_b = [], []
      for y in range(model_dim[0]):
         for x in range(model_dim[1]):
            if not vel_data[y,x] is np.ma.masked and R_gp[y,x] < a:
               if 0 < theta_gp[y,x] < np.pi/3 or 5*np.pi/3 < theta_gp[y,x] < 2*np.pi:
                  v_a.append(vel_data[y,x])
               elif 2*np.pi/3 < theta_gp[y,x] < 4*np.pi/3:
                  v_b.append(vel_data[y,x])
      if np.average(v_a) < np.average(v_b):
         pa_deg += 180
         pa_deg %= 360
         intr_vel_model = initModel(plateid, ifudesign, pa_deg, inc_deg, model_dim, 'vel', V_sys, V_max, h_rot)
         r_ip, R_gp, phi_ip, theta_gp, ellipse, bin_assign = deproject(intr_vel_model, pa_deg, inc_deg)
      
      pos_angle   = pa_deg *np.pi/180
      inclination = inc_deg *np.pi/180
      disk_param  = [R_gp, theta_gp, ellipse, bin_assign, pos_angle, inclination, V_sys, phi_ip]


      values_at_bin_center = []
      for b in range(num_bins):
         values_at_bin_center.append( [ round(V_max *np.tanh(bin_centers[b]/h_rot),3), \
                                       round(sigma_cen *np.exp(-(bin_centers[b]/a)**2),3), \
                                       round(0,3), \
                                       round(0,3) \
                                       ] )

      initial_smooth = smoothing(bin_centers, values_at_bin_center, component)

   else:
      ### (10/02/2017) enable continuation of previous run from specified iteration
      intr_vel_model = np.zeros((num_chains,model_dim[0],model_dim[1]))
      V_sys          = np.zeros(num_chains)
      phi_bar_rad    = np.zeros(num_chains)
      inc_rad        = np.zeros(num_chains)
      inc_deg        = np.zeros(num_chains)
      PA_rad         = np.zeros(num_chains)
      PA_deg         = np.zeros(num_chains)
      values_at_bin_center = [[] for i in range(num_chains)]
      disk_param           = [[] for i in range(num_chains)]

      for walker in range(num_chains):
         logT = np.log10(temp_ladder[walker])
         modelfile = fits.open(testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/'%(walker)+'manga_'+plateid+'_'+ifudesign+'_logT=%.2f'%(logT)+'_last_intr_vel_model.fits')
         intr_vel_model[walker] = modelfile[0].data
         modelfile.close()

         if walker == 0:
            tracker = open(testdir+plateid+'-'+ifudesign+'/'+component+'/tracker/walker%.0f'%(walker)+'/manga_'+plateid+'_'+ifudesign+'_logT=%.2f_it%04d.dat'%(logT,last_iteration),'r')
         else:
            tracker = open(testdir+plateid+'-'+ifudesign+'/'+component+'/tracker/walker%.0f'%(walker)+'/manga_'+plateid+'_'+ifudesign+'_logT=%.2f_last_iteration.dat~'%(logT),'r')

         last_line = tracker.readlines()[-1].split()
         tracker.close()

         V_sys[walker]       = float(last_line[6])
         phi_bar_rad[walker] = float(last_line[7])
         inc_rad[walker]     = float(last_line[9])
         inc_deg[walker]     = float(last_line[10])
         PA_rad[walker]      = float(last_line[11])
         PA_deg[walker]      = float(last_line[12])
         
         for b in range(num_bins):
            tuple_of_parameters = []
            for p in range(4):
               tuple_of_parameters.append(float(last_line[13+p+4*b]))
            values_at_bin_center[walker].append(tuple_of_parameters)

         r_ip, R_gp, phi_ip, theta_gp, ellipse, bin_assign = deproject(intr_vel_model[walker], pa_deg, inc_deg[walker])
         disk_param[walker] = [R_gp, theta_gp, ellipse, bin_assign, PA_rad[walker], inc_rad[walker], V_sys[walker], phi_ip]

      initial_smooth = smoothing(bin_centers, values_at_bin_center[0], component)

   if last_iteration is None:
      if component == 'stellar': conv_vel_model = convolveFFT(intr_vel_model*flux.data,    psf, vel_data.mask) * conv_norm
      else:                      conv_vel_model = convolveFFT(intr_vel_model*flux[0].data, psf, vel_data.mask) * conv_norm[0]
      print(np.shape(conv_vel_model))

      fits.writeto(modeldir+'vel_intr_model_PA='+str(pa_deg)+'_inc='+str(inc_deg)+'_mcmc_initial.fits',intr_vel_model,overwrite=True)
      fits.writeto(modeldir+'vel_conv_model_PA='+str(pa_deg)+'_inc='+str(inc_deg)+'_mcmc_initial.fits',np.ma.filled(conv_vel_model,np.nan),overwrite=True)

   else:
      if component == 'stellar': conv_vel_model = convolveFFT(intr_vel_model[0]*flux.data,    psf, vel_data.mask) * conv_norm
      else:                      conv_vel_model = convolveFFT(intr_vel_model[0]*flux[0].data, psf, vel_data.mask) * conv_norm[0]

   disp_data, disp_ivar = getData(plateid, ifudesign, component, 'disp')
   
   if last_iteration is None:
      intr_disp_model = initModel(plateid, ifudesign, pa_deg, inc_deg, model_dim, ext='disp', sigma=sigma_cen)
   else:
      ### (10/02/2017) enable continuation of previous run from specified iteration
      intr_disp_model = np.zeros((num_chains,model_dim[0],model_dim[1]))

      for walker in range(num_chains):
         logT = np.log10(temp_ladder[walker])
         modelfile = fits.open(testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/'%(walker)+'manga_'+plateid+'_'+ifudesign+'_logT=%.2f'%(logT)+'_last_intr_disp_model.fits')
         intr_disp_model[walker] = modelfile[0].data
         modelfile.close()
   
   if last_iteration is None:
      vel_avg = convolveFFT(intr_vel_model, psf/np.sum(psf), None)
      if component == 'stellar':
         conv_disp_model = np.sqrt( convolveFFT((intr_disp_model**2+(intr_vel_model-vel_avg)**2)*flux.data, psf, disp_data.mask) * conv_norm )
      else:
         conv_disp_model = np.sqrt( convolveFFT((intr_disp_model**2+(intr_vel_model-vel_avg)**2)*flux[1].data, psf, disp_data.mask) * conv_norm[1] )
      fits.writeto(modeldir+'disp_intr_model_PA='+str(pa_deg)+'_inc='+str(inc_deg)+'_mcmc_initial.fits',intr_disp_model,overwrite=True)
      fits.writeto(modeldir+'disp_conv_model_PA='+str(pa_deg)+'_inc='+str(inc_deg)+'_mcmc_initial.fits',np.ma.filled(conv_disp_model,np.nan),overwrite=True)
      
      print('  ### PA (deg) = %.1f, inclination (deg) = %.1f' % (pa_deg, inc_deg))

   else:
      vel_avg = convolveFFT(intr_vel_model[0], psf/np.sum(psf), None)
      if component == 'stellar':
         conv_disp_model = np.sqrt( convolveFFT((intr_disp_model[0]**2+(intr_vel_model[0]-vel_avg)**2)*flux.data, psf, disp_data.mask) * conv_norm )
      else:
         conv_disp_model = np.sqrt( convolveFFT((intr_disp_model[0]**2+(intr_vel_model[0]-vel_avg)**2)*flux[1].data, psf, disp_data.mask) * conv_norm[1] )

      print('  ### PA (deg) = %.1f, inclination (deg) = %.1f' % (pa_deg, inc_deg[0]))

   vel_chi2, vel_dim = computeChiSquared(vel_data, vel_ivar, np.ma.filled(conv_vel_model,np.nan))
   print('  ### initial velocity model:             chi2 = %.5e, %.0f pixels, %.0f parameters, reduced chi2 = %.7f' % (vel_chi2, vel_dim, n_params_vel+num_global_params, vel_chi2/(vel_dim-(n_params_vel+num_global_params))))
   print('  ### initial velocity smoothing term:    %.6f' % (sum(initial_smooth[0])))

   disp_chi2, disp_dim = computeChiSquared(disp_data, disp_ivar, np.ma.filled(conv_disp_model,np.nan))
   print('  ### initial dispersion model:           chi2 = %.5e, %.0f pixels, %.0f parameters, reduced chi2 = %.7f' % (disp_chi2, disp_dim, n_params_disp+num_global_params, disp_chi2/(disp_dim-(n_params_disp+num_global_params))))
   print('  ### initial dispersion smoothing term:  %.6f' % (initial_smooth[1]))
   
   total_chi2 = vel_chi2 + disp_chi2
   print('  ### initial model (total):              chi2 = %.5e, %.0f pixels, %.0f parameters, reduced chi2 = %.7f' % (total_chi2, vel_dim+disp_dim, n_params_vel+n_params_disp+num_global_params, total_chi2/(vel_dim+disp_dim-(n_params_vel+n_params_disp+num_global_params))))
   print('  ### initial smoothing term (total):     %.6f' % (sum(initial_smooth[0])+initial_smooth[1]))


   if last_iteration is None:
      ### (09/22/2017) replaced starting value of phi_b=0, which is a degenerate solution, at the time we implemented delayed start of m=2 mode fitting (by 5 iterations, or ~125,000 proposed transitions)
      ### (04/06/2018) replaced 45 degrees with 22.5 degrees (see results for 8484-12703 from TACC_03-30-18)
      initial_phi_bar = np.pi/8
      
      initial_inc = inclination
      initial_PA  = pos_angle

      models = []
      for i in range(num_chains):
         models.append([[intr_vel_model, intr_disp_model], values_at_bin_center, V_sys, initial_phi_bar, initial_inc, initial_PA, disk_param])

   else:
      ### (10/02/2017) enable continuation of previous run from specified iteration
      models = []
      for i in range(num_chains):
         models.append([[intr_vel_model[i], intr_disp_model[i]], values_at_bin_center[i], V_sys[i], phi_bar_rad[i], inc_rad[i], PA_rad[i], disk_param[i]])

   walker_position = models[0][1]
   Vsys_position = models[0][2]
   phi_bar_position = round(models[0][3] *180/np.pi, 4)
   inclination_position = round(models[0][4] *180/np.pi, 4)
   PA_position = round(models[0][5] *180/np.pi, 4)

   data = (vel_data, disp_data)
   ivar = (vel_ivar, disp_ivar)

   if last_iteration is None:
      last_iter = 0
      print('initial positions', walker_position, Vsys_position, phi_bar_position, inclination_position, PA_position)
   else: 
      last_iter = last_iteration     ### this_iter = last_iter+1 in mcmcFit()
      print('last iteration (%.0f)'%last_iter, walker_position, Vsys_position, phi_bar_position, inclination_position, PA_position)

   ### (10/02/2017) enable continuation of previous run from specified iteration
   for this_iter in range(last_iter, max_iterations):
      t1,tl = time.time(),time.localtime()
      print('      ######  iteration %.0f in ptRuns() began at: '%(this_iter+1) +time.strftime("%a, %d %b %Y %H:%M:%S", tl))
      
      burn_in = True   ### (12/16/2015) always require burn-in after swaps
                       ### (09/26/2017) long since deprecated, moved to post-processing
      
      models = ptWrapper(plateid, ifudesign, component, temp_ladder, num_chains, burn_in, data, ivar, models, psf, flux, conv_norm, this_iter, verbose)
      
      iter_file = open(testdir+plateid+'-'+ifudesign+'/'+component+'/.manga_'+plateid+'_'+ifudesign+'_'+component+'_last_iter', 'w')
      iter_file.write(str(this_iter+1))
      iter_file.close()
      
      print('      ######  iteration %.0f in ptRuns() finished at: '%(this_iter+1) +time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
      tt1 = time.time()-t1
      print('      ######  iteration %.0f in ptRuns() required: %.0f hours, %.0f minutes, %.0f seconds' % ((this_iter+1),(int(tt1)/3600),(int(tt1%3600)/60),(tt1%60)))
      print('')

      gc.collect()


### (09/18/2017) impose additional smoothing term on chi-squared
### (01/24/2018) increase penalty for Ha-6564, so that relative to chi-squared the penalty is similar to that used for fits of stellar data
###              similarly, increase penalty for sigma so that relative to chi-squared the penalty is similar to that for velocity fits
### (01/29/2018) further double penalty for Ha-6564 (from 3x to 6x the values used for stellar)
def smoothing(bins, parameters, component):

   V_t, sigma_V, V_2t, V_2r = np.transpose(parameters)
   
   sd2_V_t, sd2_V_2t, sd2_V_2r = [],[],[]
   sd2_sigma_V = []

   for i in range(1, len(bins)-1):
      h = bins[i] -bins[i-1]
      sd2_V_t.append( ( (V_t[i+1] -2*V_t[i] +V_t[i-1]) / h**2 )**2 )
      sd2_V_2t.append( ( (V_2t[i+1] -2*V_2t[i] +V_2t[i-1]) / h**2 )**2 )
      sd2_V_2r.append( ( (V_2r[i+1] -2*V_2r[i] +V_2r[i-1]) / h**2 )**2 )
      sd2_sigma_V.append( ( (sigma_V[i+1] -2*sigma_V[i] +sigma_V[i-1]) / h**2 )**2 )

   if component == 'stellar':
      return (sum(sd2_V_t), sum(sd2_V_2t), sum(sd2_V_2r)), 5*sum(sd2_sigma_V)
   elif component == 'Ha-6564':
      return (6*sum(sd2_V_t), 6*sum(sd2_V_2t), 6*sum(sd2_V_2r)), 30*sum(sd2_sigma_V)



def mcmcFit(plateid, ifudesign, component, data, ivar, model, psf, flux, conv_norm, walker, T, this_iter, output, burn_in=False, verbose=False):

   t1 = time.time()
   
   if component == 'stellar': conv_vel_model = convolveFFT(model[0][0]*flux.data,    psf, data[0].mask) * conv_norm
   else:                      conv_vel_model = convolveFFT(model[0][0]*flux[0].data, psf, data[0].mask) * conv_norm[0]
   vel_chi2, vel_dim = computeChiSquared(data[0], ivar[0], np.ma.filled(conv_vel_model,np.nan))

   vel_avg = convolveFFT(model[0][0], psf/np.sum(psf), None)
   if component == 'stellar':
      conv_disp_model = np.sqrt( convolveFFT((model[0][1]**2+(model[0][0]-vel_avg)**2)*flux.data,    psf, data[1].mask) * conv_norm )
   else:
      conv_disp_model = np.sqrt( convolveFFT((model[0][1]**2+(model[0][0]-vel_avg)**2)*flux[1].data, psf, data[1].mask) * conv_norm[1] )
   disp_chi2, disp_dim = computeChiSquared(data[1], ivar[1], np.ma.filled(conv_disp_model,np.nan))
   
   if verbose:
      print('  ### walker %.0f (T = %.1f), iteration %.0f, starting model: chi^2 = %.2e, reduced chi^2 (per pixel) = %.5f'%(walker, (T), this_iter+1, chi2, chi2/(dim-num_bins)))
   
   current_model = model
   current_chi2  = (vel_chi2, disp_chi2)
   convolved_model = (conv_vel_model, conv_disp_model)

   ### (09/18/2017) add smoothing term to chi-squared
   current_smooth = smoothing(bin_centers, current_model[1], component)
   current_smoothing_term = sum(current_smooth[0]) +current_smooth[1]

   jump_count_by_param = [[0,0,0,0] for i in range(num_bins)]  ### (09/14/2017)
   accepted = [[[],[],[],[]] for i in range(num_bins)]
   rejected = [[[],[],[],[]] for i in range(num_bins)]

   ### (09/15/2017) keep track of both kinds of accepts
   accepted_better = [[[],[],[],[]] for i in range(num_bins)]
   accepted_worse  = [[[],[],[],[]] for i in range(num_bins)]
   
   for i in range(4):      ### tag on four more variables (V_sys, phi_bar, inclination, position angle)
      jump_count_by_param.append(0)
      accepted.append([])
      rejected.append([])

      ### (09/15/2017) keep track of both kinds of accepts
      accepted_better.append([])
      accepted_worse.append([])

   logT = np.log10(T)
   if this_iter > 0:
      os.system('cp -p '+testdir+plateid+'-'+ifudesign+'/'+component+'/tracker/walker%.0f'%(walker)+'/manga_'+plateid+'_'+ifudesign+'_logT=%.2f_last_iteration.dat'%(logT)+' '+testdir+plateid+'-'+ifudesign+'/'+component+'/tracker/walker%.0f'%(walker)+'/manga_'+plateid+'_'+ifudesign+'_logT=%.2f_last_iteration.dat~'%(logT))
   if walker == 0:
      walker_tracker = open(testdir+plateid+'-'+ifudesign+'/'+component+'/tracker/walker%.0f'%(walker)+'/manga_'+plateid+'_'+ifudesign+'_logT=%.2f_it%04d.dat'%(logT,this_iter+1),'w')
   else:
      walker_tracker = open(testdir+plateid+'-'+ifudesign+'/'+component+'/tracker/walker%.0f'%(walker)+'/manga_'+plateid+'_'+ifudesign+'_logT=%.2f_last_iteration.dat'%(logT),'w')
   walker_tracker.write('#  '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())+'\n')
   walker_tracker.write('#\n')
   walker_tracker.write('#    MaNGA plate ID:   '+plateid+' \n')
   walker_tracker.write('#    MaNGA IFU design: '+ifudesign+' \n')
   walker_tracker.write('#\n')
   walker_tracker.write('#    component: '+component+' \n')
   walker_tracker.write('#\n')
   walker_tracker.write('#    walker %.0f \n'%(walker))
   walker_tracker.write('#    log T = %.2f \n'%(logT))
   walker_tracker.write('#    iteration %.0f \n'%(this_iter+1))
   walker_tracker.write('#\n')
   walker_tracker.write('#    column 1:  chi^2           (total) \n')
   walker_tracker.write('#    column 2:  smoothing term  (total) \n')
   walker_tracker.write('#    column 3:  chi^2           (velocity) \n')
   walker_tracker.write('#    column 4:  smoothing term  (velocity) \n')
   walker_tracker.write('#    column 5:  chi^2           (dispersion) \n')
   walker_tracker.write('#    column 6:  smoothing term  (dispersion) \n')
   walker_tracker.write('#    column 7:  V_sys           (km/s) \n')
   walker_tracker.write('#    column 8:  phi_b           (rad) \n')
   walker_tracker.write('#    column 9:  phi_b           (deg) \n')
   walker_tracker.write('#    column 10: inclination     (rad) \n')
   walker_tracker.write('#    column 11: inclination     (deg) \n')
   walker_tracker.write('#    column 12: position angle  (rad) \n')
   walker_tracker.write('#    column 13: position angle  (deg) \n')
   walker_tracker.write('#    column 14: V_t     (R=0)   (km/s) \n')
   walker_tracker.write('#    column 15: sigma_V (R=0)   (km/s) \n')
   walker_tracker.write('#    column 16: V_2t    (R=0)   (km/s) \n')
   walker_tracker.write('#    column 17: V_2r    (R=0)   (km/s) \n')
   walker_tracker.write('#    column 18... (etc.) \n')
   walker_tracker.write('#\n')
   walker_tracker.write('#\n')

   random_seed  = np.random.uniform(0, 1, max_int)
   random_order = np.random.randint(0, 2*num_bins+4, max_int)

   jump_count = 0
   counter = 0
   while (min(jump_count_by_param) < swap_int) and (counter < max_int):
   
      selected_bin = random_order[counter]
      
      if selected_bin == 2*num_bins:   ### fitting V_sys
         new_model, proposed_value = vary_V_sys(current_model, sigma_Vsys)
      
      elif selected_bin == 2*num_bins+1:    ### fitting phi_bar
         if this_iter > 19:
            new_model, proposed_value = vary_phi_bar(current_model, sigma_phiBar)
         else:       ### do nothing
            new_model, proposed_value = [current_model[0][0], current_model[0][1]], current_model[3]
      
      elif selected_bin == 2*num_bins+2:    ### fitting inclination
         if this_iter > 39:   ### (03/20/2018)   99:
            new_model, proposed_value, proposed_param_updates = vary_inclination(current_model, sigma_inc)
         else:       ### do nothing
            new_model, proposed_value, proposed_param_updates = \
               [current_model[0][0], current_model[0][1]], current_model[4], [current_model[6][0], current_model[6][1], current_model[6][3]]

      elif selected_bin == 2*num_bins+3:   ### fitting disk position angle
         new_model, proposed_value, proposed_param_updates = vary_PA(current_model, sigma_PA)

      else:
         radial_bin = selected_bin /2
         if selected_bin %2 == 1:    ### fitting sigma_V(R)
            new_model, proposed_value = vary_sigma_V(current_model, radial_bin, sigma_disp)
         
         else:    ### fitting (V_t(R), V_2t(R), V_2r(R) as a tuple
            if this_iter > 19:
               new_model, proposed_V_t, proposed_V_2t, proposed_V_2r = vary_V(current_model, radial_bin, sigma_V_t, sigma_V_2t, sigma_V_2r)
            else:
               new_model, proposed_V_t = vary_V_t(current_model, radial_bin, sigma_V_t)
               proposed_V_2t, proposed_V_2r = current_model[1][radial_bin][2:4]
               

      vel_avg = convolveFFT(new_model[0], psf/np.sum(psf), None)
      if component == 'stellar':
         new_conv_vel_model  = convolveFFT(new_model[0]*flux.data, psf, data[0].mask) * conv_norm
         new_conv_disp_model = np.sqrt( convolveFFT((new_model[1]**2+(new_model[0]-vel_avg)**2)*flux.data, psf, data[1].mask) * conv_norm )
      else:
         new_conv_vel_model  = convolveFFT(new_model[0]*flux[0].data, psf, data[0].mask) * conv_norm[0]
         new_conv_disp_model = np.sqrt( convolveFFT((new_model[1]**2+(new_model[0]-vel_avg)**2)*flux[1].data, psf, data[1].mask) * conv_norm[1] )
      new_conv_model = (new_conv_vel_model, new_conv_disp_model)
      
      candidate_vel_chi2,  vel_dim  = computeChiSquared(data[0], ivar[0], np.ma.filled(new_conv_vel_model,np.nan))
      candidate_disp_chi2, disp_dim = computeChiSquared(data[1], ivar[1], np.ma.filled(new_conv_disp_model,np.nan))
      candidate_chi2 = (candidate_vel_chi2, candidate_disp_chi2)

      ### (09/18/2017) add smoothing term to chi-squared
      if selected_bin < 2*num_bins:
         new_radial_param = np.copy(current_model[1])
         if selected_bin%2 == 1:
            new_radial_param[selected_bin/2][1] = proposed_value
         ### (09/19/2017) fitting (V_t, V_2t, V_2r)(r) as tuple, but keep track of them separately
         else:
            new_radial_param[selected_bin/2][0] = proposed_V_t
            new_radial_param[selected_bin/2][2] = proposed_V_2t
            new_radial_param[selected_bin/2][3] = proposed_V_2r
         candidate_smooth = smoothing(bin_centers, new_radial_param, component)

      else: candidate_smooth = current_smooth
      candidate_smoothing_term = sum(candidate_smooth[0]) +candidate_smooth[1]

      if verbose: print('')

      if (sum(candidate_chi2) +candidate_smoothing_term) < (sum(current_chi2) +current_smoothing_term):

         if selected_bin >= 2*num_bins:
            accepted[num_bins+selected_bin%(2*num_bins)].append(proposed_value)
            jump_count_by_param[num_bins+selected_bin%(2*num_bins)] += 1

            ### (09/15/2017) keep track of both kinds of accepts
            accepted_better[num_bins+selected_bin%(2*num_bins)].append(proposed_value)

         if verbose: print('  ### %.2f s since beginning, proposed jump accepted, previous reduced chi2 = %.5f, candidate reduced chi2 = %.5f'%(time.time()-t0, current_chi2/(dim-num_bins), candidate_chi2/(dim-num_bins)))

         current_model[0] = new_model
         convolved_model = new_conv_model

         if selected_bin == 2*num_bins:                 ### if fitting V_sys this iteration
            current_model[2] = current_model[6][6] = proposed_value
         elif selected_bin == 2*num_bins+1:             ### if fitting phi_bar this iteration
            current_model[3] = proposed_value
         elif selected_bin == 2*num_bins+2:             ### if fitting inclination this iteration
            current_model[4] = current_model[6][5] = proposed_value
            current_model[6][0] = proposed_param_updates[0]   ### update R_gp
            current_model[6][1] = proposed_param_updates[1]   ### update theta_gp
            current_model[6][3] = proposed_param_updates[2]   ### update bin_assign
         elif selected_bin == 2*num_bins+3:             ### if fitting disk position angle this iteration
            current_model[5] = current_model[6][4] = proposed_value
            current_model[6][0] = proposed_param_updates[0]   ### update R_gp
            current_model[6][1] = proposed_param_updates[1]   ### update theta_gp
            current_model[6][3] = proposed_param_updates[2]   ### update bin_assign
         else:                                        ### if fitting parameter with R-dependence this iteration
            if selected_bin%2 == 1:
               current_model[1][selected_bin/2][1] = proposed_value
               accepted[selected_bin/2][1].append(proposed_value)
               jump_count_by_param[selected_bin/2][1] += 1

               ### (09/15/2017) keep track of both kinds of accepts
               accepted_better[selected_bin/2][1].append(proposed_value)
           
            ### (09/19/2017) fitting (V_t, V_2t, V_2r)(r) as tuple, but keep track of them separately
            else:
               current_model[1][selected_bin/2][0] = proposed_V_t
               current_model[1][selected_bin/2][2] = proposed_V_2t
               current_model[1][selected_bin/2][3] = proposed_V_2r
               accepted[selected_bin/2][0].append(proposed_V_t)
               accepted[selected_bin/2][2].append(proposed_V_2t)
               accepted[selected_bin/2][3].append(proposed_V_2r)
               jump_count_by_param[selected_bin/2][0] += 1
               jump_count_by_param[selected_bin/2][2] += 1
               jump_count_by_param[selected_bin/2][3] += 1
               
               ### (09/15/2017) keep track of both kinds of accepts
               accepted_better[selected_bin/2][0].append(proposed_V_t)
               accepted_better[selected_bin/2][2].append(proposed_V_2t)
               accepted_better[selected_bin/2][3].append(proposed_V_2r)
               

         change_in_chi2 = (candidate_chi2[0]-current_chi2[0], candidate_chi2[1]-current_chi2[1])
         current_chi2 = candidate_chi2
         current_smooth = candidate_smooth      ### (09/18/2017)
         jump_count += 1

         ### (09/20/2017)
         walker_tracker.write('  %.3f'    % sum(current_chi2))
         walker_tracker.write('  %.4f   ' % (sum(current_smooth[0])+current_smooth[1]))
         walker_tracker.write('  %.3f'    % current_chi2[0])
         walker_tracker.write('  %.4f   ' % sum(current_smooth[0]))
         walker_tracker.write('  %.3f'    % current_chi2[1])
         walker_tracker.write('  %.4f   ' % current_smooth[1])
         walker_tracker.write('  %.3f   ' % current_model[2])              ### 2 is V_sys
         walker_tracker.write('  %.5f'    % current_model[3])              ### 3 is phi_bar (rad)
         walker_tracker.write('  %.3f   ' % (current_model[3] *180/np.pi))
         walker_tracker.write('  %.5f'    % current_model[4])              ### 4 is inclination (rad)
         walker_tracker.write('  %.3f   ' % (current_model[4] *180/np.pi))
         walker_tracker.write('  %.5f'    % current_model[5])              ### 5 is position angle (rad)
         walker_tracker.write('  %.3f   ' % (current_model[5] *180/np.pi))
         for b in range(num_bins):
            for i in range(len(current_model[1][b])):
               walker_tracker.write('  %.3f' % current_model[1][b][i])
            walker_tracker.write('   ')
         walker_tracker.write('\n')

      else:
         
         prob_accept = np.exp((sum(current_chi2) +current_smoothing_term -sum(candidate_chi2) -candidate_smoothing_term) /T)
         
         if prob_accept > random_seed[counter]:       ### accept proposed jump
            if selected_bin >= 2*num_bins:            ### if fitting global parameter (no R-dependence) this iteration
               accepted[num_bins+selected_bin%(2*num_bins)].append(proposed_value)
               jump_count_by_param[num_bins+selected_bin%(2*num_bins)] += 1
            
               ### (09/15/2017) keep track of both kinds of accepts
               accepted_worse[num_bins+selected_bin%(2*num_bins)].append(proposed_value)
            
            if verbose: print('  ### %.2f s since beginning, proposed jump ACCEPTED (prob_accept = %.5f), previous reduced chi2 = %.5f, candidate reduced chi2 = %.5f'%(time.time()-t0, prob_accept, current_chi2/(dim-num_bins), candidate_chi2/(dim-num_bins)))
            
            current_model[0] = new_model
            convolved_model = new_conv_model
            
            if selected_bin == 2*num_bins:              ### if fitting V_sys this iteration
               current_model[2] = current_model[6][6] = proposed_value
            elif selected_bin == 2*num_bins+1:          ### if fitting phi_bar this iteration
               current_model[3] = proposed_value
            elif selected_bin == 2*num_bins+2:          ### if fitting inclination this iteration
               current_model[4] = current_model[6][5] = proposed_value
               current_model[6][0] = proposed_param_updates[0]   ### update R_gp
               current_model[6][1] = proposed_param_updates[1]   ### update theta_gp
               current_model[6][3] = proposed_param_updates[2]   ### update bin_assign
            elif selected_bin == 2*num_bins+3:             ### if fitting disk position angle this iteration
               current_model[5] = current_model[6][4] = proposed_value
               current_model[6][0] = proposed_param_updates[0]   ### update R_gp
               current_model[6][1] = proposed_param_updates[1]   ### update theta_gp
               current_model[6][3] = proposed_param_updates[2]   ### update bin_assign
            else:                                     ### if fitting parameter with R-dependence this iteration
               if selected_bin%2 == 1:
                  current_model[1][selected_bin/2][1] = proposed_value
                  accepted[selected_bin/2][1].append(proposed_value)
                  jump_count_by_param[selected_bin/2][1] += 1
            
                  ### (09/15/2017) keep track of both kinds of accepts
                  accepted_worse[selected_bin/2][1].append(proposed_value)
               
               ### (09/19/2017) fitting (V_t, V_2t, V_2r)(r) as tuple, but keep track of them separately
               else:
                  current_model[1][selected_bin/2][0] = proposed_V_t
                  current_model[1][selected_bin/2][2] = proposed_V_2t
                  current_model[1][selected_bin/2][3] = proposed_V_2r
                  accepted[selected_bin/2][0].append(proposed_V_t)
                  accepted[selected_bin/2][2].append(proposed_V_2t)
                  accepted[selected_bin/2][3].append(proposed_V_2r)
                  jump_count_by_param[selected_bin/2][0] += 1
                  jump_count_by_param[selected_bin/2][2] += 1
                  jump_count_by_param[selected_bin/2][3] += 1

                  ### (09/15/2017) keep track of both kinds of accepts
                  accepted_worse[selected_bin/2][0].append(proposed_V_t)
                  accepted_worse[selected_bin/2][2].append(proposed_V_2t)
                  accepted_worse[selected_bin/2][3].append(proposed_V_2r)


            change_in_chi2 = (candidate_chi2[0]-current_chi2[0], candidate_chi2[1]-current_chi2[1])
            current_chi2 = candidate_chi2
            current_smooth = candidate_smooth      ### (09/18/2017)
            jump_count += 1

            ### (09/20/2017)
            walker_tracker.write('  %.3f'    % sum(current_chi2))
            walker_tracker.write('  %.4f   ' % (sum(current_smooth[0])+current_smooth[1]))
            walker_tracker.write('  %.3f'    % current_chi2[0])
            walker_tracker.write('  %.4f   ' % sum(current_smooth[0]))
            walker_tracker.write('  %.3f'    % current_chi2[1])
            walker_tracker.write('  %.4f   ' % current_smooth[1])
            walker_tracker.write('  %.3f   ' % current_model[2])              ### 2 is V_sys
            walker_tracker.write('  %.5f'    % current_model[3])              ### 3 is phi_bar (rad)
            walker_tracker.write('  %.3f   ' % (current_model[3] *180/np.pi))
            walker_tracker.write('  %.5f'    % current_model[4])              ### 4 is inclination (rad)
            walker_tracker.write('  %.3f   ' % (current_model[4] *180/np.pi))
            walker_tracker.write('  %.5f'    % current_model[5])              ### 5 is position angle (rad)
            walker_tracker.write('  %.3f   ' % (current_model[5] *180/np.pi))
            for b in range(num_bins):
               for i in range(len(current_model[1][b])):
                  walker_tracker.write('  %.3f' % current_model[1][b][i])
               walker_tracker.write('   ')
            walker_tracker.write('\n')
         
         else:
            if selected_bin >= 2*num_bins:            ### if fitting global parameter (no R-dependence) this iteration
               rejected[num_bins+selected_bin%(2*num_bins)].append(proposed_value)
            else:                                     ### if fitting parameter with R-dependence this iteration
               if selected_bin%2 == 1:
                  rejected[selected_bin/2][1].append(proposed_value)
            
               ### (09/19/2017) fitting (V_t, V_2t, V_2r)(r) as tuple, but keep track of them separately
               else:
                  rejected[selected_bin/2][0].append(proposed_V_t)
                  rejected[selected_bin/2][2].append(proposed_V_2t)
                  rejected[selected_bin/2][3].append(proposed_V_2r)
            
            change_in_chi2 = (np.nan, np.nan)
            if verbose: print('  ### %.2f s since beginning, proposed jump REJECTED (prob_accept = %.5f), previous reduced chi2 = %.5f, candidate reduced chi2 = %.5f'%(time.time()-t0, prob_accept, current_chi2/(dim-num_bins), candidate_chi2/(dim-num_bins)))

      counter += 1
      if verbose and walker == 0:
         print('  walker %.0f (T = %.1f), iteration %.0f-%.0f' % (walker,T,this_iter+1,counter))
         print('    chi2 = (%.3f, %.3f), change in chi2 = (%.7f, %.7f)' % (current_chi2[0],current_chi2[1],change_in_chi2[0],change_in_chi2[1]))
         print('    (%.0f, %.0f) valid pixels, reduced chi2 = (%.8f, %.8f)' % (vel_dim,disp_dim,current_chi2[0]/(vel_dim-(n_params_vel+num_global_params)),current_chi2[1]/(disp_dim-(n_params_disp+num_global_params))))
         print(jump_count_by_param[:num_bins//2], jump_count_by_param[num_bins//2:num_bins], jump_count_by_param[num_bins])
         print(current_model[1][:num_bins//2], current_model[1][num_bins//2:], current_model[2])
         print('')
   
      if verbose:
         print('  ### walker %.0f (T = %.1f), iteration %.0f, radial bin: %.1f <= R < %.1f, %.0f accepted jumps in this bin'%(walker,(T),counter,bin_edges[selected_bin][0],bin_edges[selected_bin][1],jump_count_by_param[selected_bin]))

      if verbose: print('')

   ### (07/05/2018) debug
   #print('  ### ('+plateid+'-'+ifudesign+') walker %.0f, log T = %.2f, %.0f iterations, proposed jump acceptance rate: %.5f'%(walker,logT,counter,track_accept/float(track_accept+track_reject)))

   walker_tracker.close()
   if walker == 0:
      os.system('cp '+testdir+plateid+'-'+ifudesign+'/'+component+'/tracker/walker%.0f'%(walker)+'/manga_'+plateid+'_'+ifudesign+'_logT=%.2f_it%04d.dat'%(logT,this_iter+1)+' '+testdir+plateid+'-'+ifudesign+'/'+component+'/tracker/walker%.0f'%(walker)+'/manga_'+plateid+'_'+ifudesign+'_logT=%.2f_last_iteration.dat'%(logT))

   fits.writeto(testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/'%(walker)+'manga_'+plateid+'_'+ifudesign+'_logT=%.2f_last_R_gp.fits'%(logT),      current_model[6][0],overwrite=True)
   fits.writeto(testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/'%(walker)+'manga_'+plateid+'_'+ifudesign+'_logT=%.2f_last_theta_gp.fits'%(logT),  current_model[6][1],overwrite=True)
   fits.writeto(testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/'%(walker)+'manga_'+plateid+'_'+ifudesign+'_logT=%.2f_last_bin_assign.fits'%(logT),current_model[6][3],overwrite=True)

   for i in range(2):
      try: os.system('mv '+testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/'%(walker)+'manga_'+plateid+'_'+ifudesign+'_logT=%.2f_last_intr_'%(logT)+('vel','disp')[i]+'_model.fits '+testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/'%(walker)+'manga_'+plateid+'_'+ifudesign+'_logT=%.2f_last_intr_'%(logT)+('vel','disp')[i]+'_model.fits~')
      finally: pass
      fits.writeto(testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/'%(walker)+'manga_'+plateid+'_'+ifudesign+'_logT=%.2f_last_intr_'%(logT)+('vel','disp')[i]+'_model.fits',current_model[0][i],overwrite=True)

      hdr = fits.Header()
      hdr['chi2']  = current_chi2[i]
      hdr['dim']   = (vel_dim,disp_dim)[i]
      hdr['n_bins'] = num_bins
      if i == 0:
         hdr['n_params'] = n_params_vel+num_global_params
         hdr['rchi2'] = current_chi2[i]/(vel_dim-(n_params_vel+num_global_params))
         hdr['smooth0']  = current_smooth[i][0]
         hdr['smooth1'] = current_smooth[i][1]
         hdr['smooth2'] = current_smooth[i][2]
      elif i == 1:
         hdr['n_params'] = num_bins
         hdr['rchi2'] = current_chi2[i]/(disp_dim-(n_params_disp+num_global_params))
         hdr['smooth'] = current_smooth[i]
      try: os.system('mv '+testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/'%(walker)+'manga_'+plateid+'_'+ifudesign+'_logT=%.2f_last_conv_'%(logT)+('vel','disp')[i]+'_model.fits '+testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/'%(walker)+'manga_'+plateid+'_'+ifudesign+'_logT=%.2f_last_conv_'%(logT)+('vel','disp')[i]+'_model.fits~')
      finally: pass
      fits.writeto(testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/'%(walker)+'manga_'+plateid+'_'+ifudesign+'_logT=%.2f_last_conv_'%(logT)+('vel','disp')[i]+'_model.fits',np.ma.filled(convolved_model[i],np.nan),header=hdr,overwrite=True)

   if verbose: print('')

   accept_rates_by_bin = open(testdir+plateid+'-'+ifudesign+'/'+component+'/acceptance_rates/walker%.0f'%(walker)+'/manga_'+plateid+'_'+ifudesign+'_logT=%.2f_it%04d.dat'%(logT,this_iter+1),'w')
   accept_rates_by_bin.write('#  '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())+'\n')
   accept_rates_by_bin.write('#\n')
   accept_rates_by_bin.write('#    MaNGA plate ID:   '+plateid+' \n')
   accept_rates_by_bin.write('#    MaNGA IFU design: '+ifudesign+' \n')
   accept_rates_by_bin.write('#\n')
   accept_rates_by_bin.write('#    walker %.0f \n'%(walker))
   accept_rates_by_bin.write('#    log T = %.2f \n'%(logT))
   accept_rates_by_bin.write('#    iteration %.0f \n'%(this_iter+1))
   accept_rates_by_bin.write('#\n')
   accept_rates_by_bin.write('#                      ------- accepted ------ rejected \n')
   accept_rates_by_bin.write('#                       better   worse   total          acceptance rate      (-2sigma)    (-1sigma)     median      (+1sigma)     (+2sigma) \n')

   accept_rates = []
   track_accept = 0     ### track cumulative
   track_reject = 0

   ### (09/20/2017) still 4*num_bins despite above changes because four parameters with radial dependence are tracked in separate arrays
   for b in range(num_radial_params*num_bins+4):   ### (10/03/2017)
      
      if walker == 0:  ### (01/25/2018)
         if b == num_radial_params*num_bins:              ### V_sys
            accepted_models = open(testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f'%(walker)+'/V_sys/manga_'+plateid+'_'+ifudesign+'_V_sys_accepted_models_logT=%.2f_it%04d.dat'%(logT,this_iter+1),'w')
            accepted_models.write('#  V_sys (km/s)     \n')
         
         elif b == num_radial_params*num_bins+1:          ### phi_bar
            accepted_models = open(testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f'%(walker)+'/phi_bar/manga_'+plateid+'_'+ifudesign+'_phi_bar_accepted_models_logT=%.2f_it%04d.dat'%(logT,this_iter+1),'w')
            accepted_models.write('#  phi_bar (rad) \n')

         elif b == num_radial_params*num_bins+2:          ### inclination
            accepted_models = open(testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f'%(walker)+'/incl/manga_'+plateid+'_'+ifudesign+'_incl_accepted_models_logT=%.2f_it%04d.dat'%(logT,this_iter+1),'w')
            accepted_models.write('#  inclination (rad) \n')

         elif b == num_radial_params*num_bins+3:          ### PA
            accepted_models = open(testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f'%(walker)+'/PA/manga_'+plateid+'_'+ifudesign+'_PA_accepted_models_logT=%.2f_it%04d.dat'%(logT,this_iter+1),'w')
            accepted_models.write('#  position angle (rad) \n')

         else:
            ### (09/14/2017) saving each parameter with radial dependence separately (instead of saving as a tuple)
            radial_bin = b /4
            if b %4 == 0:      ### V_t(R)
               accepted_models = open(testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f'%(walker)+'/V_t/manga_'+plateid+'_'+ifudesign+'_V_t_R=%.2f_accepted_models_logT=%.2f_it%04d.dat'%(bin_centers[b/4],logT,this_iter+1),'w')
               accepted_models.write('#  V_t (km/s) \n')
            elif b %4 == 1:    ### sigma_V(R)
               accepted_models = open(testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f'%(walker)+'/sigma_V/manga_'+plateid+'_'+ifudesign+'_sigma_V_R=%.2f_accepted_models_logT=%.2f_it%04d.dat'%(bin_centers[b/4],logT,this_iter+1),'w')
               accepted_models.write('#  sigma_V (km/s) \n')
            elif b %4 == 2:    ### V_2t(R)
               accepted_models = open(testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f'%(walker)+'/V_2t/manga_'+plateid+'_'+ifudesign+'_V_2t_R=%.2f_accepted_models_logT=%.2f_it%04d.dat'%(bin_centers[b/4],logT,this_iter+1),'w')
               accepted_models.write('#  V_2t (km/s) \n')
            elif b %4 == 3:    ### V_2r(R)
               accepted_models = open(testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f'%(walker)+'/V_2r/manga_'+plateid+'_'+ifudesign+'_V_2r_R=%.2f_accepted_models_logT=%.2f_it%04d.dat'%(bin_centers[b/4],logT,this_iter+1),'w')
               accepted_models.write('#  V_2r (km/s) \n')
            
            accepted_models.write('#  %.2f <= R < %.2f \n'%(bin_edges[b/4][0],bin_edges[b/4][1]))

         accepted_models.write('#    walker %.0f \n'%(walker))
         accepted_models.write('#    log T = %.2f \n'%(logT))
         accepted_models.write('#\n')

      if b >= num_radial_params*num_bins:
         if walker == 0:  ### (01/25/2018)
            for i in range(len(accepted[num_bins+b%num_bins])):
               accepted_models.write(' '+'%8s'%'%.5f'%(accepted[num_bins+b%num_bins][i])+'\n')
         this_bin_accepted = len(accepted[num_bins+b%num_bins])
         this_bin_rejected = len(rejected[num_bins+b%num_bins])

         ### (09/15/2017) keep track of both kinds of accepts
         this_bin_accepted_better = len(accepted_better[num_bins+b%num_bins])
         this_bin_accepted_worse  = len(accepted_worse[num_bins+b%num_bins])

      else:
         if walker == 0:  ### (01/25/2018)
            for i in range(len(accepted[b/4][b%4])):
               accepted_models.write(' '+'%8s'%'%.3f'%(accepted[b/4][b%4][i])+'\n')
         this_bin_accepted = len(accepted[b/4][b%4])
         this_bin_rejected = len(rejected[b/4][b%4])

         ### (09/15/2017) keep track of both kinds of accepts
         this_bin_accepted_better = len(accepted_better[b/4][b%4])
         this_bin_accepted_worse  = len(accepted_worse[b/4][b%4])

      this_bin_accept_rate = this_bin_accepted / float(this_bin_accepted+this_bin_rejected)

      if b >= num_radial_params*num_bins:
         accept_rates_by_bin.write('# \n')
         if   b == num_radial_params*num_bins:   accept_rates_by_bin.write('%20s'%'        V_sys (km/s)  ')
         elif b == num_radial_params*num_bins+1: accept_rates_by_bin.write('%20s'%'       phi_bar (deg)  ')
         elif b == num_radial_params*num_bins+2: accept_rates_by_bin.write('%20s'%'   inclination (deg)  ')
         elif b == num_radial_params*num_bins+3: accept_rates_by_bin.write('%20s'%'position angle (deg)  ')
         j = 1
      else:
         if b %4 == 0:
            accept_rates_by_bin.write('# \n')
            accept_rates_by_bin.write('#'+'%20s'%' %.2f <= R < %.2f '%(bin_edges[b/4][0],bin_edges[b/4][1])+'\n')
            accept_rates_by_bin.write('         V_t (km/s)   ')
         elif b %4 == 1:
            accept_rates_by_bin.write('     sigma_V (km/s)   ')
         elif b %4 == 2:
            accept_rates_by_bin.write('        V_2t (km/s)   ')
         elif b %4 == 3:
            accept_rates_by_bin.write('        V_2r (km/s)   ')
         j = 4

      accept_rates.append('%.5f'%(this_bin_accept_rate))
      track_accept += this_bin_accepted
      track_reject += this_bin_rejected

      accept_rates_by_bin.write('\t %5s'%'%.0f'%(this_bin_accepted_better)+ \
                                '\t %5s'%'%.0f'%(this_bin_accepted_worse)+ \
                                '\t %5s'%'%.0f'%(this_bin_accepted)+ \
                                '\t %5s'%'%.0f'%(this_bin_rejected)+ \
                                '\t %7s'%'%.5f'%(this_bin_accept_rate) )

      if b >= num_radial_params*num_bins:
         sample = accepted[num_bins+b%num_bins]
      else:
         sample = accepted[b/4][b%4]
   
      if len(sample) > 0:
         sample_minus_2sigma = np.percentile(sample, 2.275)
         sample_minus_1sigma = np.percentile(sample,15.865)
         sample_median       = np.median(sample)
         sample_plus_1sigma  = np.percentile(sample,84.125)
         sample_plus_2sigma  = np.percentile(sample,97.725)

         if (b > num_radial_params*num_bins):
            accept_rates_by_bin.write('\t %9s'%'%.2f'%((sample_minus_2sigma -sample_median) *180/np.pi)+ \
                                      '\t %7s'%'%.2f'%((sample_minus_1sigma -sample_median) *180/np.pi)+ \
                                      '\t %7s'%'%.2f'%(sample_median *180/np.pi)+ \
                                      '\t %7s'%'%.2f'%((sample_plus_1sigma -sample_median) *180/np.pi)+ \
                                      '\t %7s'%'%.2f'%((sample_plus_2sigma -sample_median) *180/np.pi) )

         else:
            accept_rates_by_bin.write('\t %9s'%'%.2f'%(sample_minus_2sigma -sample_median)+ \
                                      '\t %7s'%'%.2f'%(sample_minus_1sigma -sample_median)+ \
                                      '\t %7s'%'%.2f'%(sample_median)+ \
                                      '\t %7s'%'%.2f'%(sample_plus_1sigma -sample_median)+ \
                                      '\t %7s'%'%.2f'%(sample_plus_2sigma -sample_median) )
                  
      accept_rates_by_bin.write('\n')
      
      if walker == 0:  ### (01/25/2018)
         accepted_models.close()

   accept_rates_by_bin.write('#\n')
   accept_rates_by_bin.close()
   tt1 = time.time()-t1

   print('  ### ('+plateid+'-'+ifudesign+') walker %.0f, log T = %.2f, %.0f iterations, proposed jump acceptance rate: %.5f, time required: %.0f hours, %.0f minutes, %.0f seconds'%(walker,logT,counter,track_accept/float(track_accept+track_reject),(int(tt1)/3600),(int(tt1%3600)/60),(tt1%60)))

   output.put([walker, current_model, accept_rates]) #, new_ranges])



def run(plateid, ifudesign, component, verbose=False, gauss=False):

   #   plateid    = input('  Enter MaNGA plate ID:                      ')   #8485   #8082
   #   ifudesign  = input('  Enter MaNGA IFU design:                    ')   #9102   #12701
   #   iterations = input('  Enter number of swap iterations:           ')   #100  #1000
   #   gauss      = input('  Use Gaussian PSF instead? (Enter boolean): ')
   #   verbose    = input('  Verbose? (Enter boolean):                  ')
   
   try:
      os.system('mkdir '+modeldir)
      os.system('mkdir '+testdir)

      os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/')
      os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/')
      os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/acceptance_rates/')
      os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/swaps/')
      os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/sample/')
      os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/tracker/')
      for i in range(num_chains):
         os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/acceptance_rates/walker%.0f/'%(i))
         os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/tracker/walker%.0f/'%(i))
         os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/'%(i))
         if i == 0:
            os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/V_t/'%(i))
            os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/sigma_V/'%(i))
            os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/V_2t/'%(i))
            os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/V_2r/'%(i))
            os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/V_sys/'%(i))
            os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/phi_bar/'%(i))
            os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/incl/'%(i))
            os.system('mkdir '+testdir+plateid+'-'+ifudesign+'/'+component+'/sample/walker%.0f/PA/'%(i))
   finally: pass

   t0 = time.time()
   ptRuns(plateid, ifudesign, component, iterations, last_iter, verbose)

   #os.system('cp -p '+testdir+plateid+'-'+ifudesign+'/'+component+'/.manga_'+plateid+'_'+ifudesign+'_'+component+'_last_iter ' +testdir+plateid+'-'+ifudesign+'/'+component+'/.manga_'+plateid+'_'+ifudesign+'_'+component+'_last_before_switch')

   tt0 = time.time()-t0
   print('')
   print('  ### time elapsed ('+component+', MCMC): %.0f hours, %.0f minutes, %.0f seconds' % ((int(tt0)/3600),(int(tt0%3600)/60),((tt0%60))))

   import chi2_track
   chi2_track.track_chi2(plateid, ifudesign, component)

   import npp
   npp.make(plateid, ifudesign, component)

   tt0 = time.time()-t0-tt0
   print('')
   print('  ### time elapsed ('+component+', post): %.0f hours, %.0f minutes, %.0f seconds' % ((int(tt0)/3600),(int(tt0%3600)/60),((tt0%60))))



def do_galaxy(plateid, ifudesign, verbose=False):
   
   plateid, ifudesign = str(plateid), str(ifudesign)
   
   ### (03/30/2017) use GFWHM from header
   psf_hwhm = getGFWHM(plateid, ifudesign)     ### 0.5 arcsec/spaxel
   
   global num_bins, bin_centers, bin_edges
   ### (08/23/2017) use geometric mean of major and minor axes
   num_bins, bin_centers, bin_edges = binning(plateid, ifudesign, psf_hwhm)
   
   global n_params_vel, n_params_disp
   n_params_vel  = (num_radial_params-1) *num_bins
   n_params_disp = num_bins
   
   global last_iter, num_chains, intr_var, temp_ladder
   manager = mp.Manager()
   output = manager.Queue()
   proc = []
   for component in ('stellar', 'Ha-6564'):
      if last_iter is not None:
         iter_file = open(testdir+plateid+'-'+ifudesign+'/'+component+'/.manga_'+plateid+'_'+ifudesign+'_'+component+'_last_iter','r')
         last_iter = int(iter_file.readlines()[-1])
         iter_file.close()
      p = mp.Process(target=run, args=(plateid, ifudesign, component, verbose))
      proc.append(p)
      p.start()


if __name__ == '__main__':
   
   ### (09/27/2017) add intrinsic variance to denominator in chi-squared calculation
   ### (10/02/2017) enable continuation of previous run from specified iteration
   
   plateid, ifudesign = re.split('-',str(sys.argv[1]))
   last_iter = eval(sys.argv[2])
   num_chains = int(sys.argv[3])
   intr_var = float(sys.argv[4])**2
   
   if   num_chains == 2:  temp_ladder = np.logspace(0,0.2,num_chains).tolist()
   elif num_chains == 4:  temp_ladder = np.logspace(0,0.3,num_chains).tolist()
   elif num_chains == 5:  temp_ladder = np.logspace(0,0.4,num_chains).tolist()
   elif num_chains == 10: temp_ladder = np.logspace(0,0.9,num_chains).tolist()
   elif num_chains == 20: temp_ladder = np.logspace(0,0.95,num_chains).tolist()
   else: print('   no temperature ladder specified   ')
   
   sigma_V_t    = 0.3
   sigma_V_2t   = 0.2
   sigma_V_2r   = 0.2
   sigma_disp   = 0.5
   
   sigma_Vsys   = 0.1
   sigma_phiBar = 0.1  * np.pi/180
   sigma_inc    = 0.05 * np.pi/180
   sigma_PA     = 0.1  * np.pi/180

   sigma_phiBar = round(sigma_phiBar, 4)
   sigma_inc    = round(sigma_inc, 4)
   sigma_PA     = round(sigma_PA, 4)

   gauss, verbose = 0, 0

   swap = True
   
   if swap:
      swap_int = 1e30         ### minimum number of accepted jumps in each bin to advance to swap ahead of max_int
      iterations = 130        ### [deprecated] before allowing PA of m=2 mode to take on radial dependence, i.e., phi_b -> phi_b(R)
      max_int = 10000         ### maximum number of proposed jumps between swaps
   
   else:
      ### (08/03/2017) four cold chains, no swaps; this interval is for writing out data, essentially
      swap_int = 150
      iterations = 250
      max_int = 100000

   num_radial_params = 4
   num_global_params = 4

   do_galaxy(plateid, ifudesign)
