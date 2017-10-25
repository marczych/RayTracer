package(default_visibility = ["//visibility:public"])

cc_library(
    name = "objects",
    deps = [
        ":object",
        ":sphere",
        ":triangle",
    ],
)

cc_library(
    name = "object",
    srcs = ["Object.cpp"],
    hdrs = ["Object.h"],
)

cc_library(
    name = "sphere",
    srcs = ["Sphere.cpp"],
    hdrs = ["Sphere.h"],
    deps = [
        ":object",
        "//raytracer:boundaries",
        "//raytracer:intersection",
        "//raytracer:ray",
        "//raytracer:vector",
        "//raytracer/materials:material",
    ],
)

cc_library(
    name = "triangle",
    srcs = ["Triangle.cpp"],
    hdrs = ["Triangle.h"],
    deps = [
        ":object",
        "//raytracer:boundaries",
        "//raytracer:vector",
        "//raytracer:intersection",
    ],
)
