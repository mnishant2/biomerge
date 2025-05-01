#!/bin/bash
# Script for running biomedical entity experiments

set -e  # Exit on any error

function print_usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  train-ner          Train NER model with LoRA"
    echo "  train-el           Train EL model with LoRA"
    echo "  train-joint        Train joint NER+EL model with LoRA"
    echo "  merge              Merge NER and EL adapters"
    echo "  fusion             Train adapter fusion"
    echo "  eval-ner           Evaluate NER model"
    echo "  eval-el            Evaluate EL model"
    echo "  eval-joint         Evaluate joint model"
    echo "  eval-merged        Evaluate merged model on both tasks"
    echo "  sweep-ner          Run hyperparameter sweep for NER"
    echo "  sweep-el           Run hyperparameter sweep for EL"
    echo "  sweep-joint        Run hyperparameter sweep for joint model"
    echo ""
    echo "Options:"
    echo "  --alpha VALUE      Set merge alpha value (for 'merge' command)"
    echo "  --ner-path PATH    Set NER adapter path (for 'merge' and 'fusion' commands)"
    echo "  --el-path PATH     Set EL adapter path (for 'merge' and 'fusion' commands)"
    echo "  --model-path PATH  Set model path (for 'eval' commands)"
    echo "  --count N          Set number of runs for sweep commands (default: 10)"
}

# Default values
MERGE_ALPHA=0.5
NER_PATH="outputs/ner/lora_ner/best_checkpoint"
EL_PATH="outputs/el/lora_el/best_checkpoint"
MODEL_PATH=""
SWEEP_COUNT=10

# Parse command
if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

COMMAND=$1
shift

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --alpha)
            MERGE_ALPHA="$2"
            shift 2
            ;;
        --ner-path)
            NER_PATH="$2"
            shift 2
            ;;
        --el-path)
            EL_PATH="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --count)
            SWEEP_COUNT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Execute command
case $COMMAND in
    train-ner)
        echo "Training NER model with LoRA..."
        python -m src.main mode=train task=ner model=lora_ner
        ;;
    train-el)
        echo "Training EL model with LoRA..."
        python -m src.main mode=train task=el model=lora_el
        ;;
    train-joint)
        echo "Training joint NER+EL model with LoRA..."
        python -m src.main mode=train task=jointdecode model=lora_joint
        ;;
    merge)
        echo "Merging NER and EL adapters with alpha=$MERGE_ALPHA..."
        python -m src.main -c configs/merge.yaml merge_alpha=$MERGE_ALPHA \
            ner_adapter_path=$NER_PATH el_adapter_path=$EL_PATH
        ;;
    fusion)
        echo "Training adapter fusion..."
        python -m src.main -c configs/fusion.yaml \
            ner_adapter_path=$NER_PATH el_adapter_path=$EL_PATH
        ;;
    eval-ner)
        echo "Evaluating NER model..."
        if [ -n "$MODEL_PATH" ]; then
            python -m src.main -c configs/eval.yaml task=ner model_path=$MODEL_PATH
        else
            python -m src.main -c configs/eval.yaml task=ner
        fi
        ;;
    eval-el)
        echo "Evaluating EL model..."
        if [ -n "$MODEL_PATH" ]; then
            python -m src.main -c configs/eval.yaml task=el model_path=$MODEL_PATH
        else
            python -m src.main -c configs/eval.yaml task=el
        fi
        ;;
    eval-joint)
        echo "Evaluating joint model..."
        if [ -n "$MODEL_PATH" ]; then
            python -m src.main -c configs/eval.yaml task=joint model_path=$MODEL_PATH
        else
            python -m src.main -c configs/eval.yaml task=joint
        fi
        ;;
    eval-merged)
        echo "Evaluating merged model on both tasks..."
        if [ -n "$MODEL_PATH" ]; then
            python -m src.main -c configs/eval.yaml evaluation_type=merged model_path=$MODEL_PATH
        else
            echo "Error: Must specify --model-path for merged model evaluation"
            exit 1
        fi
        ;;
    sweep-ner)
        echo "Running hyperparameter sweep for NER ($SWEEP_COUNT runs)..."
        python -m scripts.run_sweep --task ner --count $SWEEP_COUNT
        ;;
    sweep-el)
        echo "Running hyperparameter sweep for EL ($SWEEP_COUNT runs)..."
        python -m scripts.run_sweep --task el --count $SWEEP_COUNT
        ;;
    sweep-joint)
        echo "Running hyperparameter sweep for joint model ($SWEEP_COUNT runs)..."
        python -m scripts.run_sweep --task joint --count $SWEEP_COUNT
        ;;
    *)
        echo "Unknown command: $COMMAND"
        print_usage
        exit 1
        ;;
esac

echo "Done!" 