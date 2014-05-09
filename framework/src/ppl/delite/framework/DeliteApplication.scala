package ppl.delite.framework

import java.io.{FileWriter, File, PrintWriter}
import scala.collection.mutable.{Map => MMap}
import scala.tools.nsc.io._
import scala.virtualization.lms.common.{BaseExp, Base}
import scala.virtualization.lms.internal.{GenericFatCodegen, ScalaCompile, GenericCodegen, ScalaCodegen, Transforming, GenerationFailedException, CCodegen, CudaCodegen}

import codegen.cpp.TargetCpp
import codegen.cuda.TargetCuda
import codegen.delite.{DeliteCodeGenPkg, DeliteCodegen, TargetDelite}
import codegen.opencl.TargetOpenCL
import codegen.scala.TargetScala
import codegen.restage.TargetRestage
import codegen.Target
import ops.DeliteOpsExp
import transform.DeliteTransform

trait DeliteApplication extends DeliteOpsExp with ScalaCompile with DeliteTransform {  
  type DeliteApplicationTarget = Target{val IR: DeliteApplication.this.type}

  /*
   * code generators
   */
  def getCodeGenPkg(t: DeliteApplicationTarget) : GenericFatCodegen{val IR: DeliteApplication.this.type}

  lazy val scalaTarget = new TargetScala{val IR: DeliteApplication.this.type = DeliteApplication.this}
  lazy val cudaTarget = new TargetCuda{val IR: DeliteApplication.this.type = DeliteApplication.this}
  lazy val cppTarget = new TargetCpp{val IR: DeliteApplication.this.type = DeliteApplication.this}
  lazy val openclTarget = new TargetOpenCL{val IR: DeliteApplication.this.type = DeliteApplication.this}
  lazy val restageTarget = new TargetRestage{val IR: DeliteApplication.this.type = DeliteApplication.this}

  def targets = {
    var target = List[DeliteApplicationTarget](scalaTarget)
    if(Config.generateCUDA)
      target = cudaTarget :: target
    if(Config.generateCpp)
      target = cppTarget :: target
    if(Config.generateOpenCL)
      target = openclTarget :: target
    target
  }
  lazy val generators: List[GenericFatCodegen{ val IR: DeliteApplication.this.type }] = targets.reverse.map(getCodeGenPkg(_))


  // TODO: refactor, this is from ScalaCompile trait
  lazy val codegen: ScalaCodegen { val IR: DeliteApplication.this.type } = 
    getCodeGenPkg(scalaTarget).asInstanceOf[ScalaCodegen { val IR: DeliteApplication.this.type }]

  // generators created by getCodeGenPkg will use the 'current' scope of the deliteGenerator as global scope
  lazy val deliteGenerator = new DeliteCodeGenPkg { val IR : DeliteApplication.this.type = DeliteApplication.this;
                                               val generators = DeliteApplication.this.generators; }

  /*
  private def setHostTargetCodegen(devicegen: GenericFatCodegen{ val IR: DeliteApplication.this.type }) = {
    generators find { _.deviceTarget == devicegen.hostTarget } match {
      case Some(hostgen) => devicegen.hostTargetCodegen = hostgen
      case _ => throw new Exception("Cannot find the host target codegen of " + devicegen.toString) 
    }
  }
  */

   
  /*
   * misc state
   */
  var args: Rep[Array[String]] = _

  var staticDataMap: Map[String,_] = _

  
  final def main(args: Array[String]) {
    println("Delite Application Being Staged:[" + this.getClass.getName + "]")

    println("******Generating the program******")

    //clean up the code gen directory
    Directory(Path(Config.buildDir)).deleteRecursively()

    val stream =
      if (Config.degFilename == ""){
        new PrintWriter(System.out)
      }
      else {
        new PrintWriter(new FileWriter(Config.degFilename))
      }

    def writeModules(baseDir: String) {
      Directory(Path(baseDir)).createDirectory()
      val writer = new FileWriter(baseDir + "modules.dm")
      writer.write("datastructures:\n")
      writer.write("kernels:datastructures\n")
      writer.close()
    }  
    
    // set transformers to be applied before codegen
    deliteGenerator.transformers = transformers
    
    //System.out.println("Staging application")
    
    deliteGenerator.emitDataStructures(Config.buildDir + File.separator)

    for (g <- generators) {
      //TODO: Remove c generator specialization
      val baseDir = Config.buildDir + File.separator + g.toString + File.separator
      writeModules(baseDir)
      g.initializeGenerator(baseDir + "kernels" + File.separator, args)
      g match {
        case gen:CCodegen => gen.headerStream.println("#include \"DeliteCpp.h\"")
        case gen:CudaCodegen => gen.headerStream.println("#include \"DeliteCuda.h\"")
        case _ =>
      }
    }

    // Generate a single source output for each generator when in debug mode
    if (Config.debug) {
      if (Config.degFilename.endsWith(".deg")) {
        for (g <- generators) {
          val streamDebug = new PrintWriter(new FileWriter(Config.degFilename.replace(".deg","." + g.toString)))
          val baseDir = Config.buildDir + File.separator + g.toString + File.separator
          g.initializeGenerator(baseDir + "kernels" + File.separator, args)
          g match {
            case gen: CCodegen => streamDebug.println("#include \"DeliteStandaloneMain.h\"\n")
            case _ => //
          }
          g.emitSource(stagedFunc, functionName, streamDebug)(fmA,fmB)
          // TODO: dot output
          reset
        }
      }
    }
    deliteGenerator.initializeGenerator(Config.buildDir, args)
    val sd = deliteGenerator.emitSource(stagedFunc, functionName, stream)(fmA,fmB)
    deliteGenerator.finalizeGenerator()

    for (g <- generators) {
      val baseDir = Config.buildDir + File.separator + g.toString + File.separator
      g.emitDataStructures(baseDir + "datastructures" + File.separator)
      g.finalizeGenerator()
    }

    if(Config.printGlobals) {
      println("Global definitions")
      for(globalDef <- globalDefs) {
        println(globalDef)
      }
    }
    
    generators foreach { _.emitTransferFunctions()}
    /*
    generators foreach { g =>
      try { g.emitTransferFunctions() } 
      catch { 
        case e: GenerationFailedException => 
        case e: Exception => throw(e) 
      }
    }
    */

    staticDataMap = Map() ++ sd map { case (s,d) => (deliteGenerator.quote(s), d) }
  }

  final def generateScalaSource(name: String, stream: PrintWriter) = {
    reset
    stream.println("object "+name+"Main {"/*}*/)
    stream.println("def main(args: Array[String]) {"/*}*/)
    stream.println("val o = new "+name)
    stream.println("o.apply(args)")
    stream.println("ppl.delite.runtime.profiler.PerformanceTimer.print(\"app\")")
    stream.println(/*{*/"}")
    stream.println(/*{*/"}")
    codegen.emitSource(liftedMain, name, stream)
  }


  final def execute(args: Array[String]) {
    println("Delite Application Being Executed:[" + this.getClass.getName + "]")

    println("******Executing the program*********")
    globalDefs = List()
    val g = compile(liftedMain)
    g(args)
  }

  /**
   * this is the entry method for our applications, user implement this method. Note, that it is missing the
   * args parameter, args are now accessed via the args field. This basically hides the notion of Reps from
   * user code
   */
  def main(): Unit
  
  /**
   * For multi-scope staging, to extract the return value of a scope
   */
  def mainWithResult(): Unit = main()
  var _mainResult: Unit = () //null // passes along whatever was returned by the block (could be staged or not staged, i.e. Rep[T] or T)
  
  def liftedMain(x: Rep[Array[String]]): Rep[Unit] = { this.args = x; val y = mainWithResult(); this._mainResult = y; this.args = null; unit(y) }
  
  /**
   * Used when staging a function (to be called by external code) rather than an entire app
  */
  def functionName = "Application"

  def registerFunction[A:Manifest,B:Manifest](func: Rep[A] => Rep[B]) = {
    stagedFunc = func
    fmA = manifest[A]
    fmB = manifest[B]
  }

  private var stagedFunc: Rep[Any] => Rep[Any] = (liftedMain _).asInstanceOf[Rep[Any] => Rep[Any]]
  private var fmA: Manifest[Any] = manifest[Array[String]].asInstanceOf[Manifest[Any]]
  private var fmB: Manifest[Any] = manifest[Unit].asInstanceOf[Manifest[Any]]

}
