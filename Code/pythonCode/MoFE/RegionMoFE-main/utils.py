import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from parse_args import args
from scipy.spatial.distance import pdist, squareform  

def load_data():
    data_path = args.data_path
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    landUse_feature = np.load(data_path + args.landUse_dist)    # NYC: (180, 11)
    landUse_feature = landUse_feature[np.newaxis]               # (1, 180, 11)
    landUse_feature = torch.Tensor(landUse_feature).to(device)

    poi_feature = np.load(data_path + args.POI_dist)            # NYC: (180, 26)
    poi_counts = np.sum(poi_feature, axis=1, keepdims=True)     # (180, 1)
    poi_counts = torch.Tensor(poi_counts).to(device)
    poi_feature = poi_feature[np.newaxis]                       # (1, 180, 26)
    poi_feature = torch.Tensor(poi_feature).to(device)

    mob_feature = np.load(data_path + args.mobility_dist)        # NYC: (180, 180)
    mob_feature = mob_feature[np.newaxis]                        # (1, 180, 180)
    mob_feature = torch.Tensor(mob_feature).to(device)

    mob_adj = np.load(data_path + args.mobility_adj)
    mob_adj = mob_adj/np.mean(mob_adj)
    mob_adj = torch.Tensor(mob_adj).to(device)

    poi_sim = np.load(data_path + args.POI_simi)
    poi_sim = torch.Tensor(poi_sim).to(device)

    land_sim = np.load(data_path + args.landUse_simi)
    land_sim = torch.Tensor(land_sim).to(device)

    features = [poi_feature, landUse_feature, mob_feature]

    return features, mob_adj, poi_sim, land_sim, poi_counts


class GeographicAdjacencyCalculator:
    
    def __init__(self, distance_type='euclidean', normalization='minmax', threshold=None):
        """
        Args:
            distance_type: ['euclidean', 'manhattan', 'haversine']
            normalization: ['minmax', 'gaussian', 'inverse']
            threshold: The areas exceeding this value indicate that they are not adjacent.
        """
        self.distance_type = distance_type
        self.normalization = normalization
        self.threshold = threshold
        self.scaler = MinMaxScaler()
    
    def load_region_coordinates(self, region_file_path):
        """     
        Args:
            region_file_path
            
        Returns:
            coordinates: [num_regions, 2] 
        """
        try:
            # load shapefile
            regions_data = np.load(region_file_path, allow_pickle=True)
            
            if regions_data.ndim == 2 and regions_data.shape[1] >= 2:
                coordinates = regions_data[:, :2]
            elif regions_data.ndim == 1:
                coordinates = self._extract_coordinates_from_structured_data(regions_data)
            else:
                raise ValueError(f"Unexpected data shape: {regions_data.shape}")
                
            print(f"Loaded {len(coordinates)} regions with coordinates")
            return coordinates.astype(np.float32)
            
        except Exception as e:
            print(f"Error loading region coordinates: {e}")
    
    def _extract_coordinates_from_structured_data(self, data):
        coordinates = []
        for item in data:
            if isinstance(item, dict):
                lat = item.get('lat', item.get('latitude', 0))
                lng = item.get('lng', item.get('longitude', 0))
                coordinates.append([lng, lat])  
            elif hasattr(item, '__len__') and len(item) >= 2:
                coordinates.append([item[0], item[1]])
            elif hasattr(item, 'centroid'):  
                try:
                    centroid = item.centroid
                    coordinates.append([centroid.x, centroid.y]) 
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            else:
                print(f"Type: {type(item)}")
        return np.array(coordinates)
    
    def haversine_distance(self, coord1, coord2):
        """
        Args:
            coord1, coord2: [lng, lat]
            
        Returns:
            distance
        """
        R = 6371  # radius of the earth 
        
        lng1, lat1 = np.radians(coord1)
        lng2, lat2 = np.radians(coord2)
        
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def compute_distance_matrix(self, coordinates):
        """  
        Args:
            coordinates: [num_regions, 2] 
            
        Returns:
            distance_matrix: [num_regions, num_regions] 
        """
        num_regions = len(coordinates)
        
        if self.distance_type == 'haversine':
            distance_matrix = np.zeros((num_regions, num_regions))
            for i in range(num_regions):
                for j in range(num_regions):
                    if i != j:
                        distance_matrix[i, j] = self.haversine_distance(
                            coordinates[i], coordinates[j]
                        )
        else:
            distances = pdist(coordinates, metric=self.distance_type)
            distance_matrix = squareform(distances)
        
        return distance_matrix
    
    def distance_to_adjacency(self, distance_matrix):
        """     
        Args:
            distance_matrix: [num_regions, num_regions] 
            
        Returns:
            adjacency_matrix: [num_regions, num_regions] 
        """
        if self.normalization == 'inverse':
            adjacency_matrix = 1.0 / (distance_matrix + 1e-6)
            np.fill_diagonal(adjacency_matrix, 0)  
            
        elif self.normalization == 'gaussian':
            sigma = np.std(distance_matrix)
            adjacency_matrix = np.exp(-(distance_matrix ** 2) / (2 * sigma ** 2))
            np.fill_diagonal(adjacency_matrix, 0)
            
        elif self.normalization == 'minmax':
            max_dist = np.max(distance_matrix)
            adjacency_matrix = 1.0 - (distance_matrix / max_dist)
            np.fill_diagonal(adjacency_matrix, 0)
            
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
        
        if self.threshold is not None:
            mask = distance_matrix > self.threshold
            adjacency_matrix[mask] = 0
        
        return adjacency_matrix
    
    def compute_adjacency_matrix(self, region_file_path):
        """
        Args:
            region_file_path
            
        Returns:
            adjacency_matrix: [num_regions, num_regions] 
        """
        coordinates = self.load_region_coordinates(region_file_path)
        
        distance_matrix = self.compute_distance_matrix(coordinates)
        
        adjacency_matrix = self.distance_to_adjacency(distance_matrix)
        
        print(f"Generated adjacency matrix with shape: {adjacency_matrix.shape}")
        print(f"Non-zero entries: {np.count_nonzero(adjacency_matrix)}")
        print(f"Adjacency range: [{np.min(adjacency_matrix):.4f}, {np.max(adjacency_matrix):.4f}]")
        
        return adjacency_matrix