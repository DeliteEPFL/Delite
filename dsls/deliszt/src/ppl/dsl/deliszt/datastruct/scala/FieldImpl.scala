package ppl.dsl.deliszt.datastruct.scala

/**
 * author: Michael Wu (mikemwu@stanford.edu)
 * last modified: 04/24/2011
 *
 * Pervasive Parallelism Laboratory (PPL)
 * Stanford University
 */

object FieldImpl {
  def apply[MO<:MeshObj:Manifest,VT:Manifest]() = {
    if(manifest[MO] <:< manifest[Cell]) {
       new FieldImpl(new Array[VT](Mesh.mesh.ncells))
    }
    else if(manifest[MO] <:< manifest[Face]) {
      new FieldImpl(new Array[VT](Mesh.mesh.nfaces))
    }
    else if(manifest[MO] <:< manifest[Edge]) {
      new FieldImpl(new Array[VT](Mesh.mesh.nedges))
    }
    else if(manifest[MO] <:< manifest[Vertex]) {
      new FieldImpl(new Array[VT](Mesh.mesh.nvertices))
    }
  }
}

class FieldImpl[MO<:MeshObj:Manifest, VT:Manifest](data : Array[VT]) extends Field[MO,VT] {
  def apply(e: MO) : VT = data(e.internalId)
  def update(e: MO, v : VT) = {
    data(e.internalId) = v
  }

  def size = data.length
  def dcApply(idx: Int) = data(idx)
  def dcUpdate(idx: Int, x: VT) = {
    data(idx) = x
  }
}