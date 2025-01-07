

# Check if two arguments are provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 testcase"
fi

inputFile="../testcases/$1.in"
answerFile="../testcases/$1.out"
argTmp="../testcases/.tmp.out"
argTime="../testcases/.tmp.time"
program=("hybrid_v1" "hybrid_v3" "hybrid_v4" "pthread_v1")


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

make clean
make

echo "============== make finished =============="

for p in "${program[@]}";
do
    echo -n "$p"
    argExe="./final_$p"
    srun -n2 -c2 time -f "%e" -o $argTime $argExe $inputFile $argTmp
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
done

make clean
