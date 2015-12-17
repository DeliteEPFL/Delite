package ppl.delite.framework.analysis

import java.io.{PrintWriter, ByteArrayOutputStream}
import scala.lms.internal._
import ppl.delite.framework.DeliteApplication
import scala.lms.util.OverloadHack

class MockStream extends ByteArrayOutputStream { 
   override def flush() {}
   override def close() {}
   def print(line: String) {}
}

//TODO: a lot of this is deprecated with the new LMS traversal framework
trait TraversalAnalysis extends GenericFatCodegen with OverloadHack {
  val IR: Expressions with Effects with FatExpressions
  import IR._
  implicit val mockStream: PrintWriter = new PrintWriter(new MockStream())
  val className: String
  var _result: Option[Any] = None

  def traverseNode(sym: Sym[Any], a: Def[Any]) = withStream(mockStream)(emitNode(sym, a))
  def traverseBlock(b: Block[Any])(implicit o: Overloaded1) = withStream(mockStream)(emitBlock(b))
  def traverse[A:Typ,B:Typ](f: Exp[A] => Exp[B]) = { emitSource(f, className, mockStream); result }
  def emitValDef(sym: Sym[Any], rhs: String) {}
  def emitAssignment(lhs: String, rhs: String) {}
  def result: Option[Any] = _result
  
  def emitSource[A : Typ](args: List[Sym[_]], body: Block[A], className: String, stream: PrintWriter): List[(Sym[Any], Any)] = {
    traverseBlock(body)
    Nil
  }
}
