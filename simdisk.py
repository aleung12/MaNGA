### last changed: 08/21/2017

from astropy.io import fits
import numpy as np
import os, time, gc, sys, types
from dirs import *


def mkdisk(pos_angle_deg,inclination_deg,ext,dim,V_sys=0.,V_max=220.,h_rot=10.,sigma_cen=250.):

   pos_angle   = pos_angle_deg   *np.pi/180
   inclination = inclination_deg *np.pi/180

   r_ip  = np.zeros((dim,dim))
   R_gp  = np.zeros((dim,dim))
   phi_ip = np.zeros((dim,dim))
   theta_gp = np.zeros((dim,dim))
   image = np.zeros((dim,dim))
   cen_x = np.shape(image)[1]//2
   cen_y = np.shape(image)[0]//2

   a = 0.5 *0.8 *dim
   b = a * np.cos(inclination)
   
   if 0 <= pos_angle < 1.5*np.pi: alpha = pos_angle + 0.5*np.pi
   else: alpha = pos_angle % (0.5*np.pi)

   ### for each image pixel, calculate radius r and azimuthal angle phi in image plane
   for y in range(np.shape(image)[0]):
      for x in range(np.shape(image)[1]):

         r = np.sqrt( (x-cen_x)**2 +(y-cen_y)**2 )
         ### azimuthal angle in image plane
         if (x == cen_x) and (y == cen_y):   phi = pos_angle
         elif (x <= cen_x) and (y >= cen_y): phi = np.arctan2(y-cen_y,x-cen_x) -0.5*np.pi
         else:                               phi = np.arctan2(y-cen_y,x-cen_x) +1.5*np.pi
         
         ### azimuthal angle theta in galaxy plane
         if (x == cen_x) and (y == cen_y):
            theta = 0.5 * np.pi
         elif (pos_angle <= phi <= pos_angle+0.5*np.pi):
            theta = np.arctan( np.tan(phi-pos_angle) *np.cos(inclination) )
         elif ((pos_angle <= 0.5*np.pi) and (pos_angle+0.5*np.pi < phi <= pos_angle+1.5*np.pi)) \
            or ((pos_angle > 0.5*np.pi) and ((pos_angle+0.5*np.pi < phi < 2*np.pi) or (0 <= phi <= pos_angle-0.5*np.pi))):
            theta = np.arctan( np.tan(phi-pos_angle) *np.cos(inclination) ) + np.pi
         else:
            theta = np.arctan( np.tan(phi-pos_angle) *np.cos(inclination) ) + 2*np.pi

         r_ip[y,x] = r
         phi_ip[y,x] = phi
         theta_gp[y,x] = theta

         sin_alpha = np.sin(alpha)
         cos_alpha = np.cos(alpha)
         X = x-cen_x
         Y = y-cen_y

         ### (square of) radial coordinate in galaxy plane (ellipse de-projected) normalized to disk radius R
         p = (X*cos_alpha +Y*sin_alpha)**2 /a**2 + (X*sin_alpha -Y*cos_alpha)**2 /b**2

         ### radius in galaxy plane
         R = a * p**0.5
         R_gp[y,x] = R

         if True:    #p <= 1:    ### truncate after convolution (02/27/17)
            if ext == 'vel':
               image[y,x] = V_sys + V_max *np.sin(inclination) *np.tanh(R/h_rot) *np.cos(theta)
            
            elif ext == 'disp':
               image[y,x] = sigma_cen * np.exp(-p)

   writedir = modeldir
   
   #print writedir+'PA='+str(pos_angle_deg)+'_i='+str(inclination_deg)+'_'+str(ext)+'disk.fits'
   fits.writeto(writedir+'PA='+str(pos_angle_deg)+'_i='+str(inclination_deg)+'_'+str(ext)+'disk.fits',image,overwrite=True)
   fits.writeto(writedir+'PA='+str(pos_angle_deg)+'_i='+str(inclination_deg)+'_'+'tanh.fits',np.tanh(R_gp/h_rot),overwrite=True)
   if not ext == 'disp':
      fits.writeto(writedir+'PA='+str(pos_angle_deg)+'_i='+str(inclination_deg)+'_'+'R_gp.fits',R_gp,overwrite=True)
      fits.writeto(writedir+'PA='+str(pos_angle_deg)+'_i='+str(inclination_deg)+'_'+'r_im.fits',r_ip,overwrite=True)
      fits.writeto(writedir+'PA='+str(pos_angle_deg)+'_i='+str(inclination_deg)+'_'+'theta_gp.fits',theta_gp,overwrite=True)
      fits.writeto(writedir+'PA='+str(pos_angle_deg)+'_i='+str(inclination_deg)+'_'+'phi_im.fits',phi_ip,overwrite=True)


if __name__ == '__main__':

   PA_deg  = [45, 135] #[0, 5, 15, 30, 45, 60, 75, 90, 120, 150, 175, 180]
   inc_deg = [-135, -45, 45, 135]   #1, 2, 3, 4, 5, 15, 30, 45, 60, 75, 85, 95, 105, 120, 135, 150, 165, 175, 180]
   exts = ['vel'] #,'disp']

   for PA in PA_deg:
      for inc in inc_deg:
         for ext in exts:
            mkdisk(PA,inc,ext)
            print('  ### PA (degrees)          = '+str(PA))
            print('  ### inclination (degrees) = '+str(inc))
            print('  ### time now:    '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
            print('')
