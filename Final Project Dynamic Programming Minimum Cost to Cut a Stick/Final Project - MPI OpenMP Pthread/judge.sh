#!/bin/bash

# Function to show usage information
usage() {
    echo "Usage: $0 program testcase"
    exit 1
}

# Check if two arguments are provided
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

set -x
# 使用 mpirun 啟動程式，並以 GNU time 計時
/usr/bin/time -f "%e" -o $argTime mpirun -np 2 $argExe $inputFile $argTmp
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
# module load icc openmpi
# ./judge.sh final_b t1000