from bing_image_downloader import downloader
downloader.download("motorcycle",limit=10000,output_dir='bike_images',adult_filter_off=True,force_replace=False)
downloader.download("car",limit=10000,output_dir='car_images',adult_filter_off=True,force_replace=False)
