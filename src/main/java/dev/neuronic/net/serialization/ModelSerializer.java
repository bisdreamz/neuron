package dev.neuronic.net.serialization;

import dev.neuronic.net.NeuralNet;
import com.github.luben.zstd.ZstdOutputStream;
import com.github.luben.zstd.ZstdInputStream;

import java.io.*;
import java.nio.file.Path;
import java.util.function.Consumer;

/**
 * High-performance neural network model serialization with compression.
 * 
 * Features:
 * - Zstd compression for smaller files
 * - Streaming I/O to handle large models
 * - Parallel preparation with sequential writing
 * - Progress callbacks for long operations
 * - Version-aware format for compatibility
 */
public class ModelSerializer {
    
    // Compression level: 1=fast, 22=max compression, 3=good balance
    private static final int COMPRESSION_LEVEL = 3;
    
    /**
     * Save a neural network model to file with compression.
     * 
     * @param model the neural network to save
     * @param filePath path to save the model
     * @param progressCallback optional progress callback (0.0 to 1.0)
     * @throws IOException if saving fails
     */
    public static void save(NeuralNet model, Path filePath, Consumer<Double> progressCallback) throws IOException {
        try (FileOutputStream fileOut = new FileOutputStream(filePath.toFile());
             BufferedOutputStream buffered = new BufferedOutputStream(fileOut, 64 * 1024);
             ZstdOutputStream zstdOut = new ZstdOutputStream(buffered, COMPRESSION_LEVEL);
             DataOutputStream out = new DataOutputStream(zstdOut)) {
            
            // Write file header
            writeHeader(out);
            
            // Save the model
            model.writeTo(out, SerializationConstants.CURRENT_VERSION);
            
            // Write end marker
            out.writeInt(SerializationConstants.SECTION_END);
            
            if (progressCallback != null) {
                progressCallback.accept(1.0);
            }
        }
    }
    
    /**
     * Save a model without progress tracking.
     */
    public static void save(NeuralNet model, Path filePath) throws IOException {
        save(model, filePath, null);
    }
    
    /**
     * Load a neural network model from file.
     * 
     * @param filePath path to the model file
     * @return the loaded neural network
     * @throws IOException if loading fails
     */
    public static NeuralNet load(Path filePath) throws IOException {
        try (FileInputStream fileIn = new FileInputStream(filePath.toFile());
             BufferedInputStream buffered = new BufferedInputStream(fileIn, 64 * 1024);
             ZstdInputStream zstdIn = new ZstdInputStream(buffered);
             DataInputStream in = new DataInputStream(zstdIn)) {
            
            // Read and validate header
            validateHeader(in);
            
            // Load the model
            NeuralNet model = NeuralNet.deserialize(in, SerializationConstants.CURRENT_VERSION);
            
            // Validate end marker
            int endMarker = in.readInt();
            if (endMarker != SerializationConstants.SECTION_END) {
                throw new IOException("Invalid file format: missing end marker");
            }
            
            return model;
        }
    }
    
    /**
     * Get the estimated file size for a model (uncompressed).
     */
    public static long estimateFileSize(NeuralNet model) {
        return 16 + model.getSerializedSize(SerializationConstants.CURRENT_VERSION); // Header + model
    }
    
    private static void writeHeader(DataOutputStream out) throws IOException {
        out.writeInt(SerializationConstants.MAGIC_NUMBER);
        out.writeInt(SerializationConstants.CURRENT_VERSION);
        out.writeLong(System.currentTimeMillis()); // Timestamp
    }
    
    private static void validateHeader(DataInputStream in) throws IOException {
        int magic = in.readInt();
        if (magic != SerializationConstants.MAGIC_NUMBER) {
            throw new IOException("Invalid file format: wrong magic number");
        }
        
        int version = in.readInt();
        if (version > SerializationConstants.CURRENT_VERSION) {
            throw new IOException("Unsupported file version: " + version + 
                                " (current version: " + SerializationConstants.CURRENT_VERSION + ")");
        }
        
        long timestamp = in.readLong(); // Read but don't validate timestamp
    }
}