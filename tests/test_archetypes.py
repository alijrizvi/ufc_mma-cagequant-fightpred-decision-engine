from src.engine.archetypes import FighterArchetypeEngine
from src.engine.etl import OctagonDataEngine


def test_kmeans_archetype_clustering():
    """Enforces that fighters are correctly assigned to one of four tactical archetypes."""
    db_engine = OctagonDataEngine()
    db_engine.seed_historical_lightweight_dataset()
    
    archetype_engine = FighterArchetypeEngine(db_engine)
    archetype_engine.fit()

    profile = db_engine.load_fighter_profile("Islam Makhachev")
    archetype_name, cluster_id = archetype_engine.predict_archetype(profile)

    assert isinstance(archetype_name, str)
    assert 0 <= cluster_id < 4
    assert archetype_name in [
        "Master Grappler / Submission Artist",
        "Precision Counter-Sniper",
        "Volume Pressure Striker",
        "All-Round Tactical Brawler"
    ]

def test_knn_nearest_fighters():
    """Enforces that KNN retrieves top-K historical peer fighters with distance metrics."""
    db_engine = OctagonDataEngine()
    db_engine.seed_historical_lightweight_dataset()

    archetype_engine = FighterArchetypeEngine(db_engine)
    archetype_engine.fit()

    profile = db_engine.load_fighter_profile("Carlos Prates")
    neighbors = archetype_engine.get_nearest_fighters(profile, k=2)

    assert len(neighbors) == 2
    assert "fighter_name" in neighbors[0]
    assert "distance" in neighbors[0]
    assert neighbors[0]["distance"] >= 0.0