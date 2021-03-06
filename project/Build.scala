import sbt._
import Keys._

object DeliteBuild extends Build {
  val virtualization_lms_core = "EPFL" % "macro-lms_2.11" % "1.0.0-wip-macro"

  System.setProperty("showSuppressedErrors", "false")

  val virtScala = Option(System.getenv("SCALA_VIRTUALIZED_VERSION")).getOrElse("2.11.2")
  val paradiseVersion = "2.0.1"
  val virtBuildSettingsBase = Defaults.defaultSettings ++ Seq(
    organization := "stanford-ppl",
    scalaVersion := virtScala,
    publishArtifact in (Compile, packageDoc) := false,
    libraryDependencies += "org.scala-lang.virtualized" %% "scala-virtualized" % "1.0.0-macrovirt",
    libraryDependencies += virtualization_lms_core,
    libraryDependencies += "org.scala-lang" % "scala-library" % virtScala,
    libraryDependencies += "org.scala-lang" % "scala-compiler" % virtScala,
    libraryDependencies += "org.scalatest" % "scalatest_2.11" % "2.2.2",

    libraryDependencies += "org.apache.commons" % "commons-math" % "2.2",
    libraryDependencies += "com.google.protobuf" % "protobuf-java" % "2.5.0",
    libraryDependencies += "org.apache.mesos" % "mesos" % "0.20.1",
    libraryDependencies += "org.apache.hadoop" % "hadoop-common" % "2.5.1",
    libraryDependencies += "org.apache.hadoop" % "hadoop-client" % "2.5.1",
    libraryDependencies += "org.apache.hadoop" % "hadoop-hdfs" % "2.5.1",

    retrieveManaged := true,
    scalacOptions += "-Yno-generic-signatures",


    libraryDependencies ++= (
      if (scalaVersion.value.startsWith("2.10")) List("org.scalamacros" %% "quasiquotes" % paradiseVersion)
      else Nil
      ),

    libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value % "compile",

    addCompilerPlugin("org.scalamacros" % "paradise" % paradiseVersion cross CrossVersion.full)
  )

  val virtBuildSettings = virtBuildSettingsBase ++ Seq(
    scalaSource in Compile <<= baseDirectory(_ / "src"),
    scalaSource in Test <<= baseDirectory(_ / "tests"),
    parallelExecution in Test := false,
    concurrentRestrictions in Global += Tags.limitAll(1) //we need tests to run in isolation across all projects
  )

  // build targets

  //default project: just the dependencies needed to export Delite to others (e.g., Forge)
  lazy val delite = Project("delite", file("."), //root directory required to be default
    settings = virtBuildSettings) aggregate(framework, runtime, deliteTest)

  lazy val framework = Project("framework", file("framework"), settings = virtBuildSettings) dependsOn(runtime) // dependency to runtime because of Scopes
  lazy val deliteTest = Project("delite-test", file("framework/delite-test"), settings = virtBuildSettings) dependsOn(framework, runtime)

  lazy val dsls = Project("dsls", file("dsls"), settings = virtBuildSettings) aggregate(optiql)
  lazy val optiql = Project("optiql", file("dsls/optiql"), settings = virtBuildSettings) dependsOn(framework, deliteTest)

  lazy val apps = Project("apps", file("apps"), settings = virtBuildSettings) aggregate(optiqlApps)
  lazy val optiqlApps = Project("optiql-apps", file("apps/optiql"), settings = virtBuildSettings) dependsOn(optiql)

  lazy val runtime = Project("runtime", file("runtime"), settings = virtBuildSettings)

  //include all projects that should be built (dependsOn) and tested (aggregate)
  lazy val tests = (Project("tests", file("project/boot"), settings = virtBuildSettings)
    dependsOn(optiqlApps) aggregate(framework, deliteTest, optiql))
}
