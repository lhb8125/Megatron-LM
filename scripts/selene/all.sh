# lawrence mcafee

# ACCOUNT=adlr_nlp_llmnext
ACCOUNT=llmservice_dev_mcore

# MODEL_SIZE="15b" PPS="1"
MODEL_SIZE="340b" PPS='"2" "4"'
# for PP in "${PPS}"; do
for PP in "2" "4"; do
    echo "~~~~ launch m ${MODEL_SIZE}, p ${PP} ~~~~"
    sbatch \
        --export=MODEL_SIZE="${MODEL_SIZE}",PP="${PP}" \
        -A ${ACCOUNT} \
        --job-name=${ACCOUNT}-lmcafee:lmcafee_m${MODEL_SIZE}-p${PP} \
        ./single.sh
done

# eof
