package ppl.delite.framework.codegen.delite

import java.io.{File, StringWriter, PrintWriter}
import scala.virtualization.lms.internal.{GenericFatCodegen, GenerationFailedException}
import ppl.delite.framework.ops._
import ppl.delite.framework.Config


trait DeliteKernelCodegen extends GenericFatCodegen {
  val IR: DeliteOpsExp
  import IR.{ __newVar => _, __assign => _, __ifThenElse => _ , _ }

  private def vals(sym: Sym[Any]) : List[Sym[Any]] = sym match {
    case Def(Reify(s, u, effects)) => if (s.isInstanceOf[Sym[Any]]) List(s.asInstanceOf[Sym[Any]]) else Nil
    case Def(Reflect(NewVar(v), u, effects)) => Nil
    case _ => List(sym)
  }

  private def vars(sym: Sym[Any]) : List[Sym[Any]] = sym match {
    case Def(Reflect(NewVar(v), u, effects)) => List(sym)
    case _ => Nil
  }

  private def dataDeps(rhs: Any) = { // don't use getFreeVarNode...
    val bound = boundSyms(rhs)
    val used = syms(rhs)
    // println( "=== used for " + sym)
    // used foreach { s => s match {
      // case Def(x) => println(s + " = " + x)
      // case _ => println(s)
    // }}
    //println(used)
    //focusFatBlock(used) { freeInScope(bound, used) } filter { case Def(r@Reflect(x,u,es)) => used contains r; case _ => true } // distinct
    focusFatBlock(used.map(Block(_))) { freeInScope(bound, used) } // distinct
    //syms(rhs).flatMap(s => focusBlock(s) { freeInScope(boundSyms(rhs), s) } ).distinct
  }

  def inputVals(rhs: Any): List[Sym[Any]] = dataDeps(rhs) flatMap { vals(_) }
  def inputVars(rhs: Any): List[Sym[Any]] = dataDeps(rhs) flatMap { vars(_) }

  def hasOutputSlotTypes(rhs: Any) = rhs match {
    case op: AbstractLoop[_] => true
    case op: AbstractFatLoop => true
    case op: AbstractFatLoopNest => true
    case _ => false
  }

  private var kernelStreams: List[(Any,StringWriter)] = Nil
  private var level = -1

  override def emitKernel(sym: List[Sym[Any]], rhs: Any) = {
    try {
      level += 1
      if (!kernelStreams.exists(_._1 == rhs)) { //don't emit the same kernel twice
        kernelStreams ::= (rhs, new StringWriter())
        emitKernel(sym, rhs, new PrintWriter(kernelStreams.head._2))
      }

      //combine all nested kernels into a single stream
      if (level == 0) {
        emitFileHeader()
        for ((_,kstream) <- kernelStreams) {
          stream.println(kstream.toString)
          stream.println("/**********/\n")
        }
        kernelStreams = Nil
      }
      level -= 1
    } catch {
      case e: Throwable =>
        kernelStreams = Nil
        level = -1
        throw e
    }
  }

  def emitKernel(sym: List[Sym[Any]], rhs: Any, kstream: PrintWriter) = {
    val resultIsVar = rhs match { case NewVar(_) => true case _ => false }
    val external = rhs.isInstanceOf[DeliteOpExternal[_]]
    val kernelName = sym.map(quote).mkString("")
    val inVals = inputVals(rhs)
    val inVars = inputVars(rhs)

    deliteKernel = rhs match {
      case op:AbstractFatLoop => true
      case op:AbstractFatIfThenElse => true
      case op:DeliteOp[_] => true
      case _ => false
    }

    kernelInit(sym, inVals, inVars, resultIsVar)

    /**
     * we emit the body first because we want to throw GenerationFailedExceptions and skip the rest of this function asap
     * emitKernelExtra, emitHeader etc. can mutate auxiliary output streams: we don't want this to happen if the kernel can't be generated
     */
    val bodyString = new StringWriter()
    val bodyStream = new PrintWriter(bodyString)
    rhs match {
      case d: Def[Any] => withStream(bodyStream){ emitNode(sym(0), d) }
      case f: FatDef => withStream(bodyStream){ emitFatNode(sym, f) }
    }
    bodyStream.flush()

    // TODO: we only want to mangle the result type if it is a delite op AND it will be generated AS a delite
    // op (e.g. not generated by a more specific (DSL) generator).
    // this is an issue, for example, with BLAS and MatrixSigmoid which is only a DeliteOp if BLAS is off.
    // perhaps this should be a (DeliteOp, SingleTask) variant.
    // TR: if we introduce a predicate (e.g. isDeliteOp) instead of always matching on the types this would
    // be taken care of as well (plus we'd no longer need DeliteIfThenElse, DeliteWhile, ...)
    val resultType: String = (this.toString, rhs) match {
      case ("scala", op: AbstractFatLoopNest) => 
        "activation_"+kernelName
      case ("scala", op: AbstractFatLoop) => 
        "activation_"+kernelName
      case ("scala", op: AbstractFatIfThenElse) =>
        //"generated.scala.DeliteOpMultiLoop[" + "activation_"+kernelName + "]"
        //TODO: support fat if
        assert(sym.length == 1, "TODO: support fat if")
        remap(sym.head.tp)
      case ("scala", z) => z match {
        case op: AbstractLoop[_] => "activation_"+kernelName
        case _ => remap(sym.head.tp)
      }
      case ("cpp", op: AbstractFatLoop) =>
        "activation_" + kernelName
      case ("cpp", z) => z match {
        case op: AbstractLoop[_] => "activation_" + kernelName
        case _ => remap(sym.head.tp)
      }
      case ("cuda", op: AbstractFatLoop) =>
        "void"
      case ("cuda", op: AbstractFatIfThenElse) =>
        assert(sym.length == 1, "TODO: support fat if")
        "void"
      case ("cuda", z) => z match {
        case op: AbstractLoop[_] => "void"
        case _ => remap(sym.head.tp)
      }
      case ("opencl", op: AbstractFatLoop) =>
        "void"
      case ("opencl", op: AbstractFatIfThenElse) =>
        assert(sym.length == 1, "TODO: support fat if")
        "void"
      case ("opencl", z) => z match {
        case op: AbstractLoop[_] => "void"
        case _ => remap(sym.head.tp)
      }
      case ("maxj", z) => {
        z match {
          case op: AbstractLoop[_] => "void"
          case op: AbstractFatLoop => "void"
          case _ => remap(sym.head.tp)
        }
      }
      case _ =>
        println(s"[resultType] ${(this.toString, rhs)}")
        assert(sym.length == 1) // if not set hasOutputSlotTypes and use activation record
        remap(sym.head.tp)
    }

    assert(hasOutputSlotTypes(rhs) || sym.length == 1)

    // emit kernel
    if (hasOutputSlotTypes(rhs)) {
      // activation record class declaration
      rhs match {
        case d:Def[Any] =>  withStream(kstream){ emitNodeKernelExtra(sym, d) }
        case f:FatDef => withStream(kstream){ emitFatNodeKernelExtra(sym, f) }
      }
    }

    withStream(kstream){ emitKernelHeader(sym, inVals, inVars, resultType, resultIsVar, external, hasOutputSlotTypes(rhs)) }
    kstream.println(bodyString.toString)
    withStream(kstream){ emitKernelFooter(sym, inVals, inVars, resultType, resultIsVar, external, hasOutputSlotTypes(rhs)) }
  }

}
