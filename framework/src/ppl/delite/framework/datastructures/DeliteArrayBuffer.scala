package ppl.delite.framework.datastructures

import scala.lms.common._
import java.io.PrintWriter
import reflect.{SourceContext, RefinedManifest}
import ppl.delite.framework.Config
import ppl.delite.framework.ops._
import ppl.delite.framework.Util._


/*
* Pervasive Parallelism Laboratory (PPL)
* Stanford University
*
*/

trait DeliteArrayBuffer[A] extends DeliteCollection[A]

trait DeliteArrayBufferOps extends Base {

  implicit def intTyp: Typ[Int] // import
  implicit def deliteArrayBufferTyp[A:Typ]: Typ[DeliteArrayBuffer[A]]

  object DeliteArrayBuffer {
    def apply[A:Typ]()(implicit ctx: SourceContext) = darray_buffer_new(unit(16))(implicitly[Typ[A]], ctx)
    def apply[A:Typ](initSize: Rep[Int])(implicit ctx: SourceContext) = darray_buffer_new(initSize)(implicitly[Typ[A]], ctx)
    def apply[A:Typ](data: Rep[DeliteArray[A]], length: Rep[Int])(implicit ctx: SourceContext) = darray_buffer_new_imm(data, length)
    def fromFunction[A:Typ](size: Rep[Int])(func: Rep[Int] => Rep[A])(implicit ctx: SourceContext) = darray_buffer_from_function(size, func)
  }

  implicit def repDArrayBufferToDArrayBufferOps[A:Typ](b: Rep[DeliteArrayBuffer[A]]) = new DeliteArrayBufferOpsCls(b)

  class DeliteArrayBufferOpsCls[A:Typ](d: Rep[DeliteArrayBuffer[A]]) {
    def +=(elem: Rep[A])(implicit ctx: SourceContext) = darray_buffer_append(d,elem)
    def ++=(elems: Rep[DeliteArray[A]])(implicit ctx: SourceContext) = darray_buffer_appendAll(d,elems)
    def insert(pos: Rep[Int], elem: Rep[A])(implicit ctx: SourceContext) = darray_buffer_insert(d,pos,elem)
    def result(implicit ctx: SourceContext) = darray_buffer_result(d)
    def apply(idx: Rep[Int])(implicit ctx: SourceContext) = darray_buffer_apply(d,idx)
    def update(idx: Rep[Int], x: Rep[A])(implicit ctx: SourceContext) = darray_buffer_update(d,idx,x)
    def length(implicit ctx: SourceContext) = darray_buffer_length(d)
    def mutable(implicit ctx: SourceContext) = darray_buffer_mutable(d)
    def immutable(implicit ctx: SourceContext) = darray_buffer_immutable(d)

    def map[B:Typ](func: Rep[A] => Rep[B])(implicit ctx: SourceContext) = darray_buffer_map(d,func)
    def filter(pred: Rep[A] => Rep[Boolean])(implicit ctx: SourceContext) = darray_buffer_filter(d,pred)
    def zip[B:Typ,R:Typ](that: Rep[DeliteArrayBuffer[B]])(func: (Rep[A],Rep[B]) => Rep[R])(implicit ctx: SourceContext) = darray_buffer_zip(d,that,func)
    def reduce(func: (Rep[A],Rep[A]) => Rep[A])(zero: Rep[A])(implicit ctx: SourceContext) = darray_buffer_reduce(d,func,zero)
    def foreach(func: Rep[A] => Rep[Unit])(implicit ctx: SourceContext) = darray_buffer_foreach(d,func)
    def forIndices(func: Rep[Int] => Rep[Unit])(implicit ctx: SourceContext) = darray_buffer_forIndices(d,func)
    def groupBy[K:Typ](key: Rep[A] => Rep[K])(implicit ctx: SourceContext) = darray_buffer_groupBy(d,key)
    def groupByReduce[K:Typ,V:Typ](key: Rep[A] => Rep[K], value: Rep[A] => Rep[V], reduce: (Rep[V],Rep[V]) => Rep[V]) = darray_buffer_groupByReduce(d,key,value,reduce)
    def flatMap[B:Typ](func: Rep[A] => Rep[DeliteArrayBuffer[B]])(implicit ctx: SourceContext) = darray_buffer_flatmap(d,func)
  }

  def darray_buffer_new[A:Typ](initSize: Rep[Int])(implicit ctx: SourceContext): Rep[DeliteArrayBuffer[A]]
  def darray_buffer_new_imm[A:Typ](data: Rep[DeliteArray[A]], length: Rep[Int])(implicit ctx: SourceContext): Rep[DeliteArrayBuffer[A]]
  def darray_buffer_from_function[A:Typ](size: Rep[Int], func: Rep[Int] => Rep[A])(implicit ctx: SourceContext): Rep[DeliteArrayBuffer[A]]
  def darray_buffer_apply[A:Typ](d: Rep[DeliteArrayBuffer[A]], idx: Rep[Int])(implicit ctx: SourceContext): Rep[A]
  def darray_buffer_update[A:Typ](d: Rep[DeliteArrayBuffer[A]], idx: Rep[Int], x: Rep[A])(implicit ctx: SourceContext): Rep[Unit]
  def darray_buffer_length[A:Typ](d: Rep[DeliteArrayBuffer[A]])(implicit ctx: SourceContext): Rep[Int]
  def darray_buffer_append[A:Typ](d: Rep[DeliteArrayBuffer[A]], elem: Rep[A])(implicit ctx: SourceContext): Rep[Unit]
  def darray_buffer_appendAll[A:Typ](d: Rep[DeliteArrayBuffer[A]], elems: Rep[DeliteArray[A]])(implicit ctx: SourceContext): Rep[Unit]
  def darray_buffer_insert[A:Typ](d: Rep[DeliteArrayBuffer[A]], pos: Rep[Int], elem: Rep[A])(implicit ctx: SourceContext): Rep[Unit]
  def darray_buffer_result[A:Typ](d: Rep[DeliteArrayBuffer[A]])(implicit ctx: SourceContext): Rep[DeliteArray[A]]
  def darray_buffer_mutable[A:Typ](d: Rep[DeliteArrayBuffer[A]])(implicit ctx: SourceContext): Rep[DeliteArrayBuffer[A]]
  def darray_buffer_immutable[A:Typ](d: Rep[DeliteArrayBuffer[A]])(implicit ctx: SourceContext): Rep[DeliteArrayBuffer[A]]

  def darray_buffer_map[A:Typ,B:Typ](d: Rep[DeliteArrayBuffer[A]], func: Rep[A] => Rep[B])(implicit ctx: SourceContext): Rep[DeliteArrayBuffer[B]]
  def darray_buffer_filter[A:Typ](d: Rep[DeliteArrayBuffer[A]], pred: Rep[A] => Rep[Boolean])(implicit ctx: SourceContext): Rep[DeliteArrayBuffer[A]]
  def darray_buffer_zip[A:Typ,B:Typ,R:Typ](d: Rep[DeliteArrayBuffer[A]], that: Rep[DeliteArrayBuffer[B]], func: (Rep[A],Rep[B]) => Rep[R])(implicit ctx: SourceContext): Rep[DeliteArrayBuffer[R]]
  def darray_buffer_reduce[A:Typ](d: Rep[DeliteArrayBuffer[A]], func: (Rep[A],Rep[A]) => Rep[A], zero: Rep[A])(implicit ctx: SourceContext): Rep[A]
  def darray_buffer_foreach[A:Typ](d: Rep[DeliteArrayBuffer[A]], func: Rep[A] => Rep[Unit])(implicit ctx: SourceContext): Rep[Unit]
  def darray_buffer_forIndices[A:Typ](d: Rep[DeliteArrayBuffer[A]], func: Rep[Int] => Rep[Unit])(implicit ctx: SourceContext): Rep[Unit]
  def darray_buffer_groupBy[A:Typ,K:Typ](d: Rep[DeliteArrayBuffer[A]], key: Rep[A] => Rep[K])(implicit ctx: SourceContext): Rep[DeliteMap[K,DeliteArrayBuffer[A]]]
  def darray_buffer_groupByReduce[A:Typ,K:Typ,V:Typ](d: Rep[DeliteArrayBuffer[A]], key: Rep[A] => Rep[K], value: Rep[A] => Rep[V], reduce: (Rep[V],Rep[V]) => Rep[V])(implicit ctx: SourceContext): Rep[DeliteMap[K,V]]
  def darray_buffer_flatmap[A:Typ,B:Typ](d: Rep[DeliteArrayBuffer[A]], func: Rep[A] => Rep[DeliteArrayBuffer[B]])(implicit ctx: SourceContext): Rep[DeliteArrayBuffer[B]]
}

trait DeliteArrayBufferCompilerOps extends DeliteArrayBufferOps { 
  def darray_buffer_unsafe_result[A:Typ](d: Rep[DeliteArrayBuffer[A]])(implicit ctx: SourceContext): Rep[DeliteArray[A]]
}

trait DeliteArrayBufferOpsExp extends DeliteArrayBufferOps with DeliteCollectionOpsExp with DeliteStructsExp {
  this: DeliteArrayOpsExpOpt with DeliteOpsExp with DeliteMapOpsExp =>

  implicit def deliteArrayBufferTyp[A:Typ]: Typ[DeliteArrayBuffer[A]] = {
    implicit val ManifestTyp(m) = typ[A]
    manifestTyp
  }

  /////////////////////////////////
  // sequential mutable buffer ops

  case class DeliteArrayBufferNew[A:Typ](initSize: Exp[Int], logicalSize: Exp[Int])(implicit ctx: SourceContext) extends DeliteStruct[DeliteArrayBuffer[A]] {
    val elems = copyTransformedElems(scala.collection.Seq("data" -> var_new(DeliteArray[A](initSize)).e, "length" -> var_new(logicalSize).e))
    val mA = manifest[A]
  }

  case class DeliteArrayBufferNewImm[A:Typ](data: Exp[DeliteArray[A]], length: Exp[Int]) extends DeliteStruct[DeliteArrayBuffer[A]] {
    val elems = copyTransformedElems(scala.collection.Seq("data" -> data, "length" -> length))
    val mA = manifest[A]
  }

  def darray_buffer_new[A:Typ](initSize: Exp[Int])(implicit ctx: SourceContext): Exp[DeliteArrayBuffer[A]] = darray_buffer_new(initSize, unit(0))

  def darray_buffer_new[A:Typ](initSize: Exp[Int], logicalSize: Exp[Int])(implicit ctx: SourceContext): Exp[DeliteArrayBuffer[A]] = reflectMutable(DeliteArrayBufferNew(initSize, logicalSize))

  def darray_buffer_new_imm[A:Typ](data: Exp[DeliteArray[A]], length: Exp[Int])(implicit ctx: SourceContext): Exp[DeliteArrayBuffer[A]] = reflectPure(DeliteArrayBufferNewImm(data, length))

  def darray_buffer_apply[A:Typ](d: Exp[DeliteArrayBuffer[A]], idx: Exp[Int])(implicit ctx: SourceContext): Exp[A] = darray_buffer_raw_data(d).apply(idx)

  def darray_buffer_update[A:Typ](d: Exp[DeliteArrayBuffer[A]], idx: Exp[Int], x: Exp[A])(implicit ctx: SourceContext) = darray_buffer_raw_data(d).update(idx,x)

  def darray_buffer_append[A:Typ](d: Exp[DeliteArrayBuffer[A]], elem: Exp[A])(implicit ctx: SourceContext): Exp[Unit] = {
    //darray_buffer_insert(d, d.length, elem)
    darray_buffer_ensureextra(d,unit(1))
    darray_buffer_update(d,d.length,elem)
    darray_buffer_set_length(d, delite_int_plus(d.length,unit(1)))
  }

  def darray_buffer_appendAll[A:Typ](d: Exp[DeliteArrayBuffer[A]], elems: Exp[DeliteArray[A]])(implicit ctx: SourceContext): Exp[Unit] = {
    darray_buffer_insertAll(d, d.length, elems)
  }

  def darray_buffer_result[A:Typ](d: Exp[DeliteArrayBuffer[A]])(implicit ctx: SourceContext): Exp[DeliteArray[A]] = {
    val data = darray_buffer_raw_data(d)
    val result = DeliteArray[A](d.length)
    darray_copy(data, unit(0), result, unit(0), d.length)
    delite_unsafe_immutable(result)
  }

  def darray_buffer_unsafe_result[A:Typ](d: Exp[DeliteArrayBuffer[A]])(implicit ctx: SourceContext): Exp[DeliteArray[A]] = {
    darray_buffer_raw_data(d)
  }

  protected def darray_buffer_raw_data[A:Typ](d: Exp[DeliteArrayBuffer[A]])(implicit ctx: SourceContext): Exp[DeliteArray[A]] = field[DeliteArray[A]](d, "data")
  protected def darray_buffer_set_raw_data[A:Typ](d: Exp[DeliteArrayBuffer[A]], data: Exp[DeliteArray[A]]) = field_update[DeliteArray[A]](d, "data", data)

  def darray_buffer_length[A:Typ](d: Exp[DeliteArrayBuffer[A]])(implicit ctx: SourceContext) = field[Int](d, "length")
  protected def darray_buffer_set_length[A:Typ](d: Exp[DeliteArrayBuffer[A]], len: Exp[Int]) = field_update[Int](d, "length", len)

  def darray_buffer_insert[A:Typ](d: Exp[DeliteArrayBuffer[A]], pos: Exp[Int], x: Exp[A])(implicit ctx: SourceContext): Exp[Unit] = {
    darray_buffer_insertspace(d,pos,unit(1))
    darray_buffer_update(d,pos,x)
  }

  protected def darray_buffer_insertAll[A:Typ](d: Exp[DeliteArrayBuffer[A]], pos: Exp[Int], xs: Exp[DeliteArray[A]])(implicit ctx: SourceContext): Exp[Unit] = {
    darray_buffer_insertspace(d,pos,xs.length)
    darray_buffer_copyfrom(d, pos, xs)
  }

  protected def darray_buffer_copyfrom[A:Typ](d: Exp[DeliteArrayBuffer[A]], pos: Exp[Int], xs: Exp[DeliteArray[A]])(implicit ctx: SourceContext): Exp[Unit] = {
    val data = darray_buffer_raw_data(d)
    darray_copy(xs, unit(0), data, pos, xs.length)
  }

  protected def darray_buffer_insertspace[A:Typ](d: Exp[DeliteArrayBuffer[A]], pos: Exp[Int], len: Exp[Int])(implicit ctx: SourceContext): Exp[Unit] = {
    darray_buffer_ensureextra(d,len)
    val data = darray_buffer_raw_data(d)
    darray_copy(data, pos, data, delite_int_plus(pos, len), delite_int_minus(d.length, pos))
    darray_buffer_set_length(d, delite_int_plus(d.length, len))
  }

  protected def darray_buffer_ensureextra[A:Typ](d: Exp[DeliteArrayBuffer[A]], extra: Exp[Int])(implicit ctx: SourceContext): Exp[Unit] = {
    val data = darray_buffer_raw_data(d)
    if (delite_less_than(delite_int_minus(data.length, d.length), extra)) {    
      darray_buffer_realloc(d, delite_int_plus(d.length, extra))
    }
  }

  protected def darray_buffer_realloc[A:Typ](d: Exp[DeliteArrayBuffer[A]], minLen: Exp[Int])(implicit ctx: SourceContext): Exp[Unit] = {
    val oldData = darray_buffer_raw_data(d)
    val doubleLength = delite_int_times(oldData.length, unit(2))
    val n = var_new(if (delite_greater_than(unit(4), doubleLength)) unit(4) else doubleLength)
    while (delite_less_than(n, minLen)) n = delite_int_times(n, unit(2))
    val newData = DeliteArray[A](n)
    darray_copy(oldData, unit(0), newData, unit(0), d.length)
    darray_buffer_set_raw_data(d, delite_unsafe_immutable(newData))
  }

  /////////////////////
  // parallel bulk ops

  case class DeliteArrayBufferMap[A:Typ,B:Typ](in: Exp[DeliteArrayBuffer[A]], func: Exp[A] => Exp[B])(implicit ctx: SourceContext)
    extends DeliteOpMap[A,B,DeliteArrayBuffer[B]] {

    val size = copyTransformedOrElse(_.size)(in.length)
    override def alloc(len: Exp[Int]) = darray_buffer_new[B](len,len) //flat
  }

  case class DeliteArrayBufferMapIndices[A:Typ](size: Exp[Int], func: Exp[Int] => Exp[A])
    extends DeliteOpMapIndices[A,DeliteArrayBuffer[A]] {

    override def alloc(len: Exp[Int]) = darray_buffer_new[A](len,len) //flat
  }

  case class DeliteArrayBufferZipWith[A:Typ,B:Typ,R:Typ](inA: Exp[DeliteArrayBuffer[A]], inB: Exp[DeliteArrayBuffer[B]], func: (Exp[A], Exp[B]) => Exp[R])
    extends DeliteOpZipWith[A,B,R,DeliteArrayBuffer[R]] {

    override def alloc(len: Exp[Int]) = darray_buffer_new[R](len,len) //flat
    val size = copyTransformedOrElse(_.size)(inA.length)
  }

  case class DeliteArrayBufferFilter[A:Typ](in: Exp[DeliteArrayBuffer[A]], cond: Exp[A] => Exp[Boolean])
    extends DeliteOpFilter[A,A,DeliteArrayBuffer[A]] {

    override def alloc(len: Exp[Int]) = darray_buffer_new[A](len) //buffer
    val size = copyTransformedOrElse(_.size)(in.length)
    def func = e => e
  }

  case class DeliteArrayBufferFlatMap[A:Typ,B:Typ](in: Exp[DeliteArrayBuffer[A]], func: Exp[A] => Exp[DeliteArrayBuffer[B]])
    extends DeliteOpFlatMap[A,B,DeliteArrayBuffer[B]] {

    override def alloc(len: Exp[Int]) = darray_buffer_new[B](len) //buffer
    val size = copyTransformedOrElse(_.size)(in.length)
  }
  
  case class DeliteArrayBufferReduce[A:Typ](in: Exp[DeliteArrayBuffer[A]], func: (Exp[A], Exp[A]) => Exp[A], zero: Exp[A])
    extends DeliteOpReduce[A] {
    
    val size = copyTransformedOrElse(_.size)(in.length)    
  }

  case class DeliteArrayBufferGroupBy[A:Typ,K:Typ](in: Exp[DeliteArrayBuffer[A]], keyFunc: Exp[A] => Exp[K])
    extends DeliteOpGroupBy[K,A,DeliteArrayBuffer[A],DeliteArray[DeliteArrayBuffer[A]]] {

    def alloc(len: Exp[Int]) = DeliteArray[DeliteArrayBuffer[A]](len)
    def allocI(len: Exp[Int]) = darray_buffer_new[A](len) //buffer
    val size = copyTransformedOrElse(_.size)(in.length)
  }

  case class DeliteArrayBufferForeach[A:Typ](in: Exp[DeliteArrayBuffer[A]], func: Rep[A] => Rep[Unit]) extends DeliteOpForeach[A] {
    def sync = null //unused
    val size = copyTransformedOrElse(_.size)(in.length)
    val mA = manifest[A]
  }

  case class DeliteArrayBufferForIndices[A:Typ](in: Exp[DeliteArrayBuffer[A]], func: Rep[Int] => Rep[Unit])(implicit ctx: SourceContext) extends DeliteOpIndexedLoop {
    val size = copyTransformedOrElse(_.size)(in.length)
    val mA = manifest[A]
  }

  def darray_buffer_mutable[A:Typ](d: Rep[DeliteArrayBuffer[A]])(implicit ctx: SourceContext) = reflectMutable(DeliteArrayBufferMap(d,(e:Rep[A])=>e))
  def darray_buffer_immutable[A:Typ](d: Rep[DeliteArrayBuffer[A]])(implicit ctx: SourceContext) = reflectPure(DeliteArrayBufferMap(d,(e:Rep[A])=>e))

  def darray_buffer_from_function[A:Typ](size: Exp[Int], func: Exp[Int] => Exp[A])(implicit ctx: SourceContext) = reflectPure(DeliteArrayBufferMapIndices(size,func))
  def darray_buffer_map[A:Typ,B:Typ](d: Rep[DeliteArrayBuffer[A]], func: Rep[A] => Rep[B])(implicit ctx: SourceContext) = reflectPure(DeliteArrayBufferMap(d,func))
  def darray_buffer_filter[A:Typ](d: Rep[DeliteArrayBuffer[A]], pred: Rep[A] => Rep[Boolean])(implicit ctx: SourceContext) = reflectPure(DeliteArrayBufferFilter(d,pred))
  def darray_buffer_zip[A:Typ,B:Typ,R:Typ](d: Rep[DeliteArrayBuffer[A]], that: Rep[DeliteArrayBuffer[B]], func: (Rep[A],Rep[B]) => Rep[R])(implicit ctx: SourceContext) = reflectPure(DeliteArrayBufferZipWith(d,that,func))
  def darray_buffer_flatmap[A:Typ,B:Typ](d: Exp[DeliteArrayBuffer[A]], func: Rep[A] => Rep[DeliteArrayBuffer[B]])(implicit ctx: SourceContext) = reflectPure(DeliteArrayBufferFlatMap(d,func))
  def darray_buffer_reduce[A:Typ](d: Rep[DeliteArrayBuffer[A]], func: (Rep[A],Rep[A]) => Rep[A], zero: Rep[A])(implicit ctx: SourceContext) = reflectPure(DeliteArrayBufferReduce(d,func,zero))
  def darray_buffer_foreach[A:Typ](d: Rep[DeliteArrayBuffer[A]], func: Rep[A] => Rep[Unit])(implicit ctx: SourceContext) = {
    val df = DeliteArrayBufferForeach(d,func)
    reflectEffect(df, summarizeEffects(df.body.asInstanceOf[DeliteForeachElem[A]].func).star andAlso Simple())
  }
  def darray_buffer_forIndices[A:Typ](d: Rep[DeliteArrayBuffer[A]], func: Rep[Int] => Rep[Unit])(implicit ctx: SourceContext) = {
    val df = DeliteArrayBufferForIndices(d,func)
    reflectEffect(df, summarizeEffects(df.body.asInstanceOf[DeliteForeachElem[A]].func).star andAlso Simple())
  }
  def darray_buffer_groupBy[A:Typ,K:Typ](d: Exp[DeliteArrayBuffer[A]], key: Rep[A] => Rep[K])(implicit ctx: SourceContext): Exp[DeliteMap[K,DeliteArrayBuffer[A]]] = DeliteMap(d, key, reflectPure(DeliteArrayBufferGroupBy(d,key)))
  def darray_buffer_groupByReduce[A:Typ,K:Typ,V:Typ](d: Rep[DeliteArrayBuffer[A]], key: Rep[A] => Rep[K], value: Rep[A] => Rep[V], reduce: (Rep[V],Rep[V]) => Rep[V])(implicit ctx: SourceContext): Rep[DeliteMap[K,V]] = DeliteMap(d, key, value, reduce)


  /////////////////////
  // delite collection
  
  def isDeliteArrayBufferTpe(x: Typ[_])(implicit ctx: SourceContext) = isSubtype(x.erasure,classOf[DeliteArrayBuffer[_]])
  def isDeliteArrayBuffer[A](x: Exp[DeliteCollection[A]])(implicit ctx: SourceContext) = isDeliteArrayBufferTpe(x.tp)
  def asDeliteArrayBuffer[A](x: Exp[DeliteCollection[A]])(implicit ctx: SourceContext) = x.asInstanceOf[Exp[DeliteArrayBuffer[A]]]
    
  override def dc_size[A:Typ](x: Exp[DeliteCollection[A]])(implicit ctx: SourceContext) = {
    if (isDeliteArrayBuffer(x)) asDeliteArrayBuffer(x).length
    else super.dc_size(x)
  }
  
  override def dc_apply[A:Typ](x: Exp[DeliteCollection[A]], n: Exp[Int])(implicit ctx: SourceContext) = {
    if (isDeliteArrayBuffer(x)) asDeliteArrayBuffer(x).apply(n)
    else super.dc_apply(x,n)
  }
  
  override def dc_update[A:Typ](x: Exp[DeliteCollection[A]], n: Exp[Int], y: Exp[A])(implicit ctx: SourceContext) = {
    if (isDeliteArrayBuffer(x)) asDeliteArrayBuffer(x).update(n,y)
    else super.dc_update(x,n,y)        
  }
  
  override def dc_set_logical_size[A:Typ](x: Exp[DeliteCollection[A]], y: Exp[Int])(implicit ctx: SourceContext) = {
    if (isDeliteArrayBuffer(x)) {
      val buf = asDeliteArrayBuffer(x)
      darray_buffer_set_length(buf, y)
    }
    else super.dc_set_logical_size(x,y)        
  }
  
  override def dc_parallelization[A:Typ](x: Exp[DeliteCollection[A]], hasConditions: Boolean)(implicit ctx: SourceContext) = {
    if (isDeliteArrayBuffer(x)) {
      if (hasConditions) ParSimpleBuffer else ParFlat
    }
    else super.dc_parallelization(x, hasConditions)
  }

  override def dc_appendable[A:Typ](x: Exp[DeliteCollection[A]], i: Exp[Int], y: Exp[A])(implicit ctx: SourceContext) = {
    if (isDeliteArrayBuffer(x)) { unit(true) }
    else super.dc_appendable(x,i,y)
  }

  override def dc_append[A:Typ](x: Exp[DeliteCollection[A]], i: Exp[Int], y: Exp[A])(implicit ctx: SourceContext) = {
    if (isDeliteArrayBuffer(x)) { asDeliteArrayBuffer(x) += y; }
    else super.dc_append(x,i,y)
  }
  
  override def dc_alloc[A:Typ,CA<:DeliteCollection[A]:Typ](x: Exp[CA], size: Exp[Int])(implicit ctx: SourceContext): Exp[CA] = {
    if (isDeliteArrayBuffer(x)) darray_buffer_new[A](size,size).asInstanceOf[Exp[CA]] //flat alloc
    else super.dc_alloc[A,CA](x,size)
  } 
  
  override def dc_copy[A:Typ](src: Exp[DeliteCollection[A]], srcPos: Exp[Int], dst: Exp[DeliteCollection[A]], dstPos: Exp[Int], size: Exp[Int])(implicit ctx: SourceContext): Exp[Unit] = {
    if (isDeliteArrayBuffer(src) && isDeliteArrayBuffer(dst)) {
      darray_copy(darray_buffer_raw_data(asDeliteArrayBuffer(src)), srcPos, darray_buffer_raw_data(asDeliteArrayBuffer(dst)), dstPos, size)
    }
    else super.dc_copy(src,srcPos,dst,dstPos,size)
  }

  override def dc_data_field(x: Typ[_]) = {
    if (isDeliteArrayBufferTpe(x)) "data"
    else super.dc_data_field(x)
  }

  override def dc_size_field(x: Typ[_]) = {
    if (isDeliteArrayBufferTpe(x)) "length"
    else super.dc_size_field(x)
  }

  override def mirror[A:Typ](e: Def[A], f: Transformer)(implicit ctx: SourceContext): Exp[A] = (e match {
    case e@DeliteArrayBufferNewImm(d,l) => reflectPure(new {override val original = Some(f,e) } with DeliteArrayBufferNewImm(f(d),f(l))(e.mA))(mtype(manifest[A]),implicitly[SourceContext])
    case Reflect(e@DeliteArrayBufferNewImm(d,l), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with DeliteArrayBufferNewImm(f(d),f(l))(e.mA), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@DeliteArrayBufferNew(p,l), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with DeliteArrayBufferNew(f(p),f(l))(e.mA,ctx), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case e@DeliteArrayBufferMap(in,g) => reflectPure(new { override val original = Some(f,e) } with DeliteArrayBufferMap(f(in),f(g))(e.dmA,e.dmB,ctx))(mtype(manifest[A]),implicitly[SourceContext])
    case e@DeliteArrayBufferMapIndices(s,g) => reflectPure(new { override val original = Some(f,e) } with DeliteArrayBufferMapIndices(f(s),f(g))(e.dmA))(mtype(manifest[A]),implicitly[SourceContext])
    case e@DeliteArrayBufferFilter(in,g) => reflectPure(new { override val original = Some(f,e) } with DeliteArrayBufferFilter(f(in),f(g))(e.dmA))(mtype(manifest[A]),implicitly[SourceContext])
    case e@DeliteArrayBufferZipWith(a,b,g) => reflectPure(new { override val original = Some(f,e) } with DeliteArrayBufferZipWith(f(a),f(b),f(g))(e.dmA,e.dmB,e.dmR))(mtype(manifest[A]),implicitly[SourceContext])
    case e@DeliteArrayBufferReduce(in,g,z) => e.asInstanceOf[DeliteArrayBufferReduce[A]] match { //scalac typer bug
      case e@DeliteArrayBufferReduce(in,g,z) => reflectPure(new { override val original = Some(f,e) } with DeliteArrayBufferReduce(f(in),f(g),f(z))(e.dmA))(mtype(manifest[A]),implicitly[SourceContext])
    }    
    case e@DeliteArrayBufferGroupBy(in,k) => reflectPure(new { override val original = Some(f,e) } with DeliteArrayBufferGroupBy(f(in),f(k))(mtype(e.dmV),mtype(e.dmK)))(mtype(manifest[A]),implicitly[SourceContext])
    case e@DeliteArrayBufferFlatMap(in,g) => reflectPure(new { override val original = Some(f,e) } with DeliteArrayBufferFlatMap(f(in),f(g))(mtype(e.dmA),mtype(e.dmB)))(mtype(manifest[A]),implicitly[SourceContext])
    case Reflect(e@DeliteArrayBufferMap(in,g), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with DeliteArrayBufferMap(f(in),f(g))(e.dmA,e.dmB,ctx), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@DeliteArrayBufferMapIndices(s,g), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with DeliteArrayBufferMapIndices(f(s),f(g))(e.dmA), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@DeliteArrayBufferFilter(in,g), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with DeliteArrayBufferFilter(f(in),f(g))(e.dmA), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@DeliteArrayBufferZipWith(a,b,g), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with DeliteArrayBufferZipWith(f(a),f(b),f(g))(e.dmA,e.dmB,e.dmR), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@DeliteArrayBufferReduce(in,g,z), u, es) => e.asInstanceOf[DeliteArrayBufferReduce[A]] match { //scalac typer bug
      case e@DeliteArrayBufferReduce(in,g,z) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with DeliteArrayBufferReduce(f(in),f(g),f(z))(e.dmA), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    }    
    case Reflect(e@DeliteArrayBufferGroupBy(in,k), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with DeliteArrayBufferGroupBy(f(in),f(k))(mtype(e.dmV),mtype(e.dmK)), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@DeliteArrayBufferFlatMap(in,g), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with DeliteArrayBufferFlatMap(f(in),f(g))(mtype(e.dmA),mtype(e.dmB)), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@DeliteArrayBufferForeach(in,g), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with DeliteArrayBufferForeach(f(in),f(g))(mtype(e.mA)), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@DeliteArrayBufferForIndices(in,g), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with DeliteArrayBufferForIndices(f(in),f(g))(mtype(e.mA),ctx), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case _ => super.mirror(e,f)
  }).asInstanceOf[Exp[A]]

  override def unapplyStructType[T:Typ]: Option[(StructTag[T], List[(String,Typ[_])])] = manifest[T] match {
    case t if t.erasure == classOf[DeliteArrayBuffer[_]] => Some((classTag(t), List("data","length") zip List(darrayManifest(t.typeArguments(0)), manifest[Int]))) 
    case _ => super.unapplyStructType[T]
  }
}

trait ScalaGenDeliteArrayBufferOps extends ScalaGenEffect {
  val IR: DeliteArrayBufferOpsExp
}

trait CudaGenDeliteArrayBufferOps extends CudaGenEffect {
  val IR: DeliteArrayBufferOpsExp
}
