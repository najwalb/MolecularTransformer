module purge
module load gcc
module load git
module load tykky
export PATH=/appl/opt/python/3.8.14-gnu850/bin:$PATH

# This file should be run in the RetroDiffuser/installation directory directly with: bash container-run-on-puhti.sh

#Najwa: project_2006950
#Severi: project_2006174
project_name=project_2006950

#mkdir "/projappl/${project_name}/moltransformer"

#pip-containerize new --prefix /projappl/${project_name}/moltransformer torch.txt
pip-containerize update /projappl/${project_name}/moltransformer --post-install req_puhti_mahti.txt
#pip-containerize update /projappl/${project_name}/moltransformer --post-install install_this.txt

### run command below if want to use environment e.g. in interactive session
# export PATH="/projappl/project_2006950/retrodiffuser/bin:$PATH" 
# export PATH="/projappl/project_2006174/retrodiffuser/bin:$PATH"

