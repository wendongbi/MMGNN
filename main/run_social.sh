# Dataset_list:
#   * Social Networks: UGA50, GWU54, Northeastern19, Hamilton46, Caltech36, Howard90, UF21, Simmons81,  Flickr, chameleon, squirrel, Tulane29,
#   * Citation Networks: cora, pubmed, citeseer
#   * Webpage Networks: chameleon, squirrel
#   * Other Networks: Flickr, etc.
#   For more information, please refer to our paper: https://arxiv.org/pdf/2208.07012.pdf


# # For social networks
python3 main.py --dataset Simmons81 --hidden 256  --num_epoch 600 --layer 2 
# python3 main.py --dataset Hamilton46 --hidden 256  --num_epoch 600 --layer 2
# python3 main.py --dataset UGA50 --hidden 256  --num_epoch 600 --layer 2
# python3 main.py --dataset GWU54 --hidden 256  --num_epoch 600 --layer 2
# python3 main.py --dataset Howard90 --hidden 256  --num_epoch 600 --layer 2
# python3 main.py --dataset Northeastern19 --hidden 256  --num_epoch 600 --layer 2
# python3 main.py --dataset UF21 --hidden 256 --num_epoch 600 --layer 2
# python3 main.py --dataset Tulane29 --hidden 256 --num_epoch 600 --layer 2
# python3 main.py --dataset Caltech36 --hidden 256 --num_epoch 600 --layer 2


