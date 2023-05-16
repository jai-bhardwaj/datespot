package com.system.tensorhub.knn;

import java.util.Arrays;
import java.util.Objects;

/**
 * Represents a neighbor in a nearest neighbor search.
 */
public class Neighbor {
  /**
   * Gets a builder for creating a new `Neighbor` instance.
   *
   * @return A builder for creating a new `Neighbor` instance.
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Builder class for creating a `Neighbor` instance.
   */
  public static class Builder {
    /**
     * The identifier for the neighbor.
     */
    private String id;

    /**
     * Sets the identifier for the neighbor.
     *
     * @param id The identifier for the neighbor.
     * @return This builder.
     */
    public Builder withId(String id) {
      this.id = id;
      return this;
    }

    /**
     * The score for the neighbor.
     */
    private float score;

    /**
     * Sets the score for the neighbor.
     *
     * @param score The score for the neighbor.
     * @return This builder.
     */
    public Builder withScore(float score) {
      this.score = score;
      return this;
    }

    /**
     * Populates the fields of a `Neighbor` instance with the builder's fields.
     *
     * @param instance The `Neighbor` instance to populate.
     */
    protected void populate(Neighbor instance) {
      instance.setId(this.id);
      instance.setScore(this.score);
    }

    /**
     * Builds a new `Neighbor` instance.
     *
     * @return The new `Neighbor` instance.
     */
    public Neighbor build() {
      Neighbor instance = new Neighbor();

      populate(instance);

      return instance;
    }
  };

  /**
   * The identifier for the neighbor.
   */
  private String id;

  /**
   * The score for the neighbor.
   */
  private float score;

  /**
   * Gets the identifier for the neighbor.
   *
   * @return The identifier for the neighbor.
   */
  public String getId() {
    return this.id;
  }

  /**
   * Sets the identifier for the neighbor.
   *
   * @param id The identifier for the neighbor.
   */
  public void setId(String id) {
    this.id = id;
  }

  /**
   * Gets the score for the neighbor.
   *
   * @return The score for the neighbor.
   */
  public float getScore() {
    return this.score;
  }

  /**
   * Sets the score for the neighbor.
   *
   * @param score The score for the neighbor.
   */
  public void setScore(float score) {
    this.score = score;
  }

  /**
   * Hash code for the class name.
   */
  private static final int classNameHashCode =
      internalHashCodeCompute("com.system.tensorhub.knn.Neighbor");

  /**
   * Returns a hash code value for the object.
   *
   * @return A hash code value for this object.
   */
  @Override
  public int hashCode() {
    return internalHashCodeCompute(
        classNameHashCode,
        getId(),
        getScore());
  }

  /**
   * Computes the hash code for a set of objects.
   *
   * @param objects The objects for which to compute the hash code.
   * @return The hash code for the objects.
   */
  private static int internalHashCodeCompute(Object... objects) {
    return Arrays.hashCode(objects);
  }

  /**
   * Indicates whether some other object is "equal to" this one.
   *
   * @param other The reference object with which to compare.
   * @return `true` if this object is the same as the `other` argument; `false` otherwise.
   */
  @Override
  public boolean equals(final Object other) {
    if (!(other instanceof Neighbor)) {
      return false;
    }

    Neighbor that = (Neighbor) other;

    return
        Objects.equals(getId(), that.getId())
        && Objects.equals(getScore(), that.getScore());
  }
  /**
   * Returns a string representation of the Neighbor object.
   *
   * @return A string representation of the Neighbor object, in the format "Neighbor(id=<id>, score=<score>)".
   */
  @Override
  public String toString() {
    StringBuilder ret = new StringBuilder();
    ret.append("Neighbor(");

    ret.append("id=");
    ret.append(String.valueOf(id));
    ret.append(", ");

    ret.append("score=");
    ret.append(String.valueOf(score));
    ret.append(")");

    return ret.toString();
  }
}
