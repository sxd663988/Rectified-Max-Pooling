#!/bin/bash
# this script is used to batch conversion
# @author: mengxue.Zhang

function ergodic(){
  for file in `ls $1`
  do
    local path=$1"/"$file
    # echo ${1##*/}
    # echo ${file%%.*}
    if test -f $path; then
      echo $path
      if [ ! -d "./"${2}"/"${1##*/} ];then
        mkdir -p "./"${2}"/"${1##*/}
      fi
      ./mstar2jpeg -i $path -o "./"${2}"/"${1##*/}"/"${file%%.*}".jpeg" -e
    fi
  done
}

#prefix="train"
#train_dir_array=("/home/snow/new_mstar_data_10/train/2S1" "/home/snow/new_mstar_data_10/train/BMP2" "/home/snow/new_mstar_data_10/train/BRDM_2" "/home/snow/new_mstar_data_10/train/BTR60" "/home/snow/#new_mstar_data_10/train/BTR70" "/home/snow/new_mstar_data_10/train/D7" "/home/snow/new_mstar_data_10/train/T62" "/home/snow/new_mstar_data_10/train/T72" "/home/snow/new_mstar_data_10/train/ZIL131" "/home/snow/#new_mstar_data_10/train/ZSU_23_4")
#for var in ${train_dir_array[@]}
#do
  #ergodic $var $prefix
#done

#prefix="test"
#test_dir_array=("/home/snow/new_mstar_data_10/test/2S1" "/home/snow/new_mstar_data_10/test/BMP2" "/home/snow/new_mstar_data_10/test/BRDM_2" "/home/snow/new_mstar_data_10/test/BTR60" "/home/snow/#new_mstar_data_10/test/BTR70" "/home/snow/new_mstar_data_10/test/D7" "/home/snow/new_mstar_data_10/test/T62" "/home/snow/new_mstar_data_10/test/T72" "/home/snow/new_mstar_data_10/test/ZIL131" "/home/snow/#new_mstar_data_10/test/ZSU_23_4")
#for var in ${test_dir_array[@]}
#do
  #ergodic $var $prefix
#done

#prefix="eoc2"
#train_dir_array=("/home/snow/new_mstar_data_10/eoc_2/configuration_variants/T72/A32" "/home/snow/new_mstar_data_10/eoc_2/configuration_variants/T72/A62" "/home/snow/new_mstar_data_10/eoc_2/#configuration_variants/T72/A63" "/home/snow/new_mstar_data_10/eoc_2/configuration_variants/T72/A64" "/home/snow/new_mstar_data_10/eoc_2/configuration_variants/T72/S7" "/home/snow/new_mstar_data_10/eoc_2/#version_variants/BMP_2/9566" "/home/snow/new_mstar_data_10/eoc_2/version_variants/BMP_2/C21" "/home/snow/new_mstar_data_10/eoc_2/version_variants/T72/812" "/home/snow/new_mstar_data_10/eoc_2/version_variants/#T72/A04" "/home/snow/new_mstar_data_10/eoc_2/version_variants/T72/A05" "/home/snow/new_mstar_data_10/eoc_2/version_variants/T72/A07" "/home/snow/new_mstar_data_10/eoc_2/version_variants/T72/A10")

#for var in ${train_dir_array[@]}
#do
  #ergodic $var $prefix
#done
