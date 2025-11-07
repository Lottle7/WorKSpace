import argparse

parser = argparse.ArgumentParser()

# -----------------------File------------------------
parser.add_argument('--city',                 default="NY",       help='City name, can be NY or Chi')
parser.add_argument('--mobility_dist',        default='/mob_dist.npy')
parser.add_argument('--POI_dist',             default='/poi_dist.npy')
parser.add_argument('--landUse_dist',         default='/landUse_dist.npy')
parser.add_argument('--mobility_adj',         default='/mob-adj.npy')
parser.add_argument('--POI_simi',             default='/poi_simi.npy')
parser.add_argument('--landUse_simi',         default='/landUse_simi.npy')

# -----------------------Model-----------------------
parser.add_argument("--device",         type=int,    default=0)
parser.add_argument('--embedding_size', type=int,    default=144)
parser.add_argument('--epochs',         type=int,    default=3000)
parser.add_argument('--dropout',        type=float,  default=0.1)

parser.add_argument('--seed',           type=int,    default=42)
parser.add_argument('--learning_rate',  type=float,  default=0.0005)
parser.add_argument('--weight_decay',   type=float,  default=5e-4)
parser.add_argument('--num_views',      type=int,    default=3)
parser.add_argument('--hidden_dim_rw',  type=int,    default=256)
parser.add_argument('--num_layer_rw',   type=int,    default=3)
parser.add_argument('--temperature_rw', type=float,  default=1)
parser.add_argument('--triplet_margin', type=float,  default=0.8)

args = parser.parse_args()

# -----------------------City--------------------------- #

if args.city == 'NY':
    parser.add_argument('--data_path',                    default='./data_NY')
    parser.add_argument('--POI_dim',         type=int,    default=26)
    parser.add_argument('--landUse_dim',     type=int,    default=11)
    parser.add_argument('--region_num',      type=int,    default=180)
    parser.add_argument('--NO_IntraAFL',     type=int,    default=3)
    parser.add_argument('--NO_head',         type=int,    default=4)
    parser.add_argument('--NO_Graphormer', type=int,    default=3)
    parser.add_argument('--NO_Ghead',      type=int,  default=4)
    parser.add_argument('--d_prime',         type=int,    default=64)
    parser.add_argument('--c',               type=int,    default=32)
    parser.add_argument('--spatial_alpha',      type=float,  default=0.1)
    parser.add_argument('--region_file_path',     default='/regions_Manh.npy')
elif args.city == "Chi":
    parser.add_argument('--data_path',                    default='./data_Chi')
    parser.add_argument('--POI_dim',         type=int,    default=26)
    parser.add_argument('--landUse_dim',     type=int,    default=12)
    parser.add_argument('--region_num',      type=int,    default=77)
    parser.add_argument('--NO_IntraAFL',     type=int,    default=1)
    parser.add_argument('--NO_head',         type=int,    default=1)
    parser.add_argument('--NO_Graphormer', type=int,    default=3)
    parser.add_argument('--NO_Ghead',      type=int,  default=7)
    parser.add_argument('--d_prime',         type=int,    default=32)
    parser.add_argument('--c',               type=int,    default=32)
    parser.add_argument('--spatial_alpha',      type=float,  default=0.1)
    parser.add_argument('--region_file_path',     default='/regions_CHI.npy')

args = parser.parse_args()