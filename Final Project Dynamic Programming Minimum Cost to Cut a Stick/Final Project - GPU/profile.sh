#!/bin/bash


# Function to show usage information
usage() {
    echo "Usage: $0 program testcase"
    exit 1
}

# Check if three arguments are provided
if [ $# -ne 2 ]; then
    usage
fi

folderName="nsight-result"
inputFile="testcases/$2.in"
argExe="./$1"

# Check if the input file exists
if [ ! -e "$inputFile" ]; then
    echo "[ERROR] File $inputFile does not exist!"
    exit 2
fi

make clean
make $1

srun -N1 -n1 -c1 --gres=gpu:1 nsys profile --stats=true --force-overwrite=true -o $folderName/res.nsys-rep $argExe $inputFile /dev/null
# srun -N1 -n1 -c2 --gres=gpu:1 nvprof --metrics all --events all --force-overwrite --output-profile $folderName/res.nvvp $argExe $inputFile /dev/null
