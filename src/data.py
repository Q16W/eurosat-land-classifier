from torchvision.datasets import EuroSAT
ds = EuroSAT(root="data", download=True)
print(len(ds), ds.classes)
# 'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'