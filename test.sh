# python inpaint_celeba.py --model_path diffusion_model_celeba.pth \
#                             --mask_type center --mask_size 0.4 \
#                             --use_improved --limit 1000

# python inpaint_celeba.py --model_path diffusion_model_celeba.pth \
#                             --mask_type left_half --mask_size 0.4 \
#                             --use_improved --limit 1000
                            
# python inpaint_celeba.py --model_path diffusion_model_celeba.pth \
#                             --mask_type random --mask_size 0.4 \
#                             --use_improved --limit 1000

python conditional_diffusion.py --sample --num_samples 10 --class_id 0 --dataset "cifar" 
python conditional_diffusion.py --sample --num_samples 10 --class_id 1 --dataset "cifar" 
python conditional_diffusion.py --sample --num_samples 10 --class_id 2 --dataset "cifar" 
python conditional_diffusion.py --sample --num_samples 10 --class_id 3 --dataset "cifar" 
python conditional_diffusion.py --sample --num_samples 10 --class_id 4 --dataset "cifar" 
python conditional_diffusion.py --sample --num_samples 10 --class_id 5 --dataset "cifar" 
python conditional_diffusion.py --sample --num_samples 10 --class_id 6 --dataset "cifar" 
python conditional_diffusion.py --sample --num_samples 10 --class_id 7 --dataset "cifar" 
python conditional_diffusion.py --sample --num_samples 10 --class_id 8 --dataset "cifar" 
python conditional_diffusion.py --sample --num_samples 10 --class_id 9 --dataset "cifar" 