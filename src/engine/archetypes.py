from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.engine.etl import OctagonDataEngine
from src.engine.stats import FighterCombatProfile


class FighterArchetypeEngine:
    """Unsupervised analytics engine applying PCA, K-Means clustering, and KNN peer search to fighter metrics."""

    ARCHETYPE_LABELS = {
        0: "Master Grappler / Submission Artist",
        1: "Precision Counter-Sniper",
        2: "Volume Pressure Striker",
        3: "All-Round Tactical Brawler"
    }

    def __init__(self, db_engine: OctagonDataEngine):
        self.db_engine = db_engine
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=42)
        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
        
        self.is_fitted = False
        self.fighter_names: List[str] = []
        self.feature_matrix: np.ndarray = np.array([])

    def _extract_dataset(self) -> Tuple[List[str], np.ndarray]:
        """Extracts continuous feature vectors across all fighters registered in DuckDB."""
        df_profiles = self.db_engine.conn.execute("""
            SELECT fighter_name, punch_power, elbow_power, leg_kick_power, strike_hit_pct,
                   arm_block_pct, leg_block_pct, takedown_def_pct, durability_index,
                   takedown_success_pct, gnp_avg_damage, submission_conversion_pct, stamina_decay_rate
            FROM fighter_profiles
        """).df()

        names = df_profiles["fighter_name"].tolist()
        features = df_profiles.drop(columns=["fighter_name"]).values.astype(np.float32)
        return names, features

    def fit(self):
        """Fits StandardScaler, PCA, K-Means, and KNN models on DuckDB fighter profiles."""
        self.fighter_names, raw_features = self._extract_dataset()
        
        if len(raw_features) < 4:
            # Handle small seed rosters by adjusting clusters dynamically
            self.kmeans = KMeans(n_clusters=min(len(raw_features), 4), random_state=42, n_init=10)

        scaled_features = self.scaler.fit_transform(raw_features)
        self.feature_matrix = scaled_features
        
        self.pca.fit(scaled_features)
        self.kmeans.fit(scaled_features)
        self.knn.fit(scaled_features)
        
        self.is_fitted = True

    def _profile_to_vector(self, profile: FighterCombatProfile) -> np.ndarray:
        """Converts a FighterCombatProfile into a 2D scaled feature vector."""
        vec = np.array([[
            profile.punch_power, profile.elbow_power, profile.leg_kick_power, profile.strike_hit_pct,
            profile.arm_block_pct, profile.leg_block_pct, profile.takedown_def_pct, profile.durability_index,
            profile.takedown_success_pct, profile.gnp_avg_damage, profile.submission_conversion_pct, profile.stamina_decay_rate
        ]], dtype=np.float32)
        return self.scaler.transform(vec)

    def predict_archetype(self, profile: FighterCombatProfile) -> Tuple[str, int]:
        """Predicts the tactical cluster archetype for a given fighter profile."""
        if not self.is_fitted:
            self.fit()

        vec_scaled = self._profile_to_vector(profile)
        cluster_id = int(self.kmeans.predict(vec_scaled)[0])
        archetype_name = self.ARCHETYPE_LABELS.get(cluster_id, "All-Round Tactical Brawler")
        return archetype_name, cluster_id

    def get_nearest_fighters(self, profile: FighterCombatProfile, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top K statistically similar fighters using KNN distance matching."""
        if not self.is_fitted:
            self.fit()

        vec_scaled = self._profile_to_vector(profile)
        # Query k + 1 neighbors to account for self-match exclusion
        k_query = min(k + 1, len(self.fighter_names))
        distances, indices = self.knn.kneighbors(vec_scaled, n_neighbors=k_query)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            peer_name = self.fighter_names[idx]
            if peer_name != profile.name:
                results.append({
                    "fighter_name": peer_name,
                    "distance": round(float(dist), 4)
                })
            if len(results) == k:
                break

        return results