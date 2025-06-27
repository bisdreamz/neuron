package dev.neuronic.net.math;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Implementation class that contains Vector API dependencies.
 * This class is only loaded if Vector API is available.
 * DO NOT import this class directly - access through Vectorization facade.
 */
final class VectorizationImpl {
    
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    /**
     * Get the vector species for internal use.
     * Called via reflection from Vectorization class.
     */
    public static VectorSpecies<Float> getSpecies() {
        return SPECIES;
    }
    
    /**
     * Get the vector length.
     * Called via reflection from Vectorization class.
     */
    public static int getVectorLength() {
        return SPECIES.length();
    }
    
    private VectorizationImpl() {} // Prevent instantiation
}