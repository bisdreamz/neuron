package dev.neuronic.net.serialization;

import java.io.*;

/**
 * Interface for high-performance binary serialization.
 * 
 * Each class implements its own serialization logic for:
 * - Maintainability: Each class owns its data format
 * - Performance: Direct binary writing, no reflection
 * - Extensibility: Version-aware serialization
 */
public interface Serializable {
    
    /**
     * Write this object's data to the output stream.
     * Should write in a format that readFrom() can understand.
     * 
     * @param out output stream to write to
     * @param version serialization version for compatibility
     * @throws IOException if writing fails
     */
    void writeTo(DataOutputStream out, int version) throws IOException;
    
    /**
     * Read this object's data from the input stream.
     * Should read data written by writeTo() with the same version.
     * 
     * @param in input stream to read from
     * @param version serialization version for compatibility
     * @throws IOException if reading fails
     */
    void readFrom(DataInputStream in, int version) throws IOException;
    
    /**
     * Get the estimated serialized size in bytes.
     * Used for progress tracking and buffer allocation.
     * 
     * @param version serialization version
     * @return estimated size in bytes
     */
    int getSerializedSize(int version);
    
    /**
     * Get the type identifier for this serializable class.
     * Used during deserialization to know which class to instantiate.
     * 
     * @return unique type identifier
     */
    int getTypeId();
}