# Test Resources

## MNIST Dataset

The MnistEndToEndTest requires the MNIST dataset to be available in this directory.

To run the MNIST tests:

1. Copy the entire `datasets` folder from the project root to this directory:
   ```bash
   cp -r ../../../../../../datasets .
   ```

2. The structure should look like:
   ```
   src/test/resources/
   └── datasets/
       ├── DB_MNIST.data
       ├── DB_MNIST.lobs
       ├── DB_MNIST.properties
       └── DB_MNIST.script
   ```

3. Run the tests:
   ```bash
   mvn test -Dtest=MnistEndToEndTest
   ```

The test will automatically skip if the dataset is not present.