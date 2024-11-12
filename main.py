from feature_extractor import FeatureExtractor


def main():
    extractor = FeatureExtractor()
    train_features_pca, val_features_pca = extractor.process()


if __name__ == "__main__":
    main()
