# Perspective Correction

The goal of this project is to implement a simple perspective correction algorithm. The algorithm should be able to
correct the perspective of a given image, so that the resulting image looks like it was taken from the top.
The use case is that the user on the floor of a warehouse takes a picture of the box with the label on it.

Once the picture is corrected for perspective, we can use OCR to read the label on the box. Additionally, since the
label sizes are usually standard, we can also derive the distance of the label from the sides and various other
positions to gauge if the package is free of chargebacks.

Additionally, if there are errors present in the packaging of the box, we can also detect the cost of the chargebacks
associated with each package, performance of each worker, and the overall performance of the warehouse.

## How to run:

1. Clone the repository

```bash
git clone https://github.com/adwalvekar/RetailReady.git && cd RetailReady
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the following command:

```bash
python3 perspective_correction.py
```

The output will be a series of images shown one at a time. Use the 'Enter/Return' key to move to the next image.

The outputs are as follows:

1. Original Image
2. Label with detected contours
3. Perspective corrected image

![Screenshot.png](Screenshot.png)