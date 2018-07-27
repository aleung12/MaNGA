import os, sys, re

if __name__ == '__main__':

    PLT_IFU = str(sys.argv[1])
    plateid, ifudesign = re.split('-', PLT_IFU)
    if   ifudesign[:3] == '127': runtime = 16
    elif ifudesign[:2] ==  '91': runtime = 12
    elif ifudesign[:2] ==  '61': runtime = 10
    elif ifudesign[:2] ==  '37': runtime = 8
    elif ifudesign[:2] ==  '19': runtime = 6 

    slurm_file = open('fr/slurm/manga_'+PLT_IFU+'.slurm', 'w')
    slurm_file.write('#!/bin/bash\n'+
                     '#\n'+
                     '# Simple SLURM script for submitting multiple serial\n'+
                     '# jobs (e.g. parametric studies) using a script wrapper\n'+
                     '# to launch the jobs.\n'+
                     '#\n'+
                     '# To use, build the launcher executable and your\n'+
                     '# serial application(s) and place them in your WORKDIR\n'+
                     '# directory.  Then, edit the CONTROL_FILE to specify\n'+
                     '# each executable per process.\n'+
                     '\n'+
                     '#-------------------------------------------------------\n'+
                     '#-------------------------------------------------------\n'+
                     '#\n'+
                     '#------------------Scheduler Options--------------------\n'+
                     '#SBATCH -J '+PLT_IFU+'                       # Job name\n'+
                     '#SBATCH -n 1                                # Total number of tasks\n'+
                     '#SBATCH -p gpu                              # Queue name\n'+
                     '#SBATCH -o manga-c10.out.'+PLT_IFU+'.%j      # Name of stdout output file (%j expands to jobid)\n'+
                     '#SBATCH -t %.0f:00:00                         # Run time (hh:mm:ss)\n'%runtime+
                     '#SBATCH -A Dynamics-of-Galaxies\n'+
                     '\n'+
                     '\n'+
                     '#----------------\n'+
                     '# Job Submission\n'+
                     '#----------------\n'+
                     '\n'+
                     'python /home/04032/leung/manga/fmodel/MPL-6/mcmc_tacc.py '+PLT_IFU+' None 10 1 \n')
    slurm_file.close()

    os.system('cd fr/slurm; sbatch manga_'+PLT_IFU+'.slurm')
