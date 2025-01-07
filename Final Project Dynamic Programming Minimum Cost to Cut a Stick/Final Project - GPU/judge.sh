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

inputFile="testcases/$2.in"
answerFile="testcases/$2.out"

# Check if the input file exists
if [ ! -e "$inputFile" ]; then
    echo "[ERROR] File $inputFile does not exist!"
    exit 2
fi

# Check if the answer file exists
if [ ! -e "$answerFile" ]; then
    echo "[ERROR] File $answerFile does not exist!"
    exit 2
fi

# Assign arguments to variables
argExe="./$1"
argTmp="testcases/.tmp.out"
argTime="testcases/.tmp.time"

make clean
make $1

set -x
srun -N1 -n1 -c1 --gres=gpu:1 time -f "%e" -o $argTime $argExe $inputFile $argTmp
set +x

exeTime=`cat $argTime`
rm $argTime
echo 
echo -n "$exeTime: "

if cmp -s $answerFile $argTmp; then
    echo -e "\e[32mCorrect\e[0m"
else
    echo -e "\e[31mWrong\e[0m"
fi

rm $argTmp