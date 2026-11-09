// Read printed slate text off contact-sheet tiles with Apple Vision.
//
// Runs locally on this Mac: opens one image read-only, crops each
// requested grid tile, runs one accurate text request per tile, and
// prints one JSON object on stdout. It uploads nothing, launches
// nothing, writes nothing, and prints nothing but that JSON on success.
import Foundation
import Vision
import CoreGraphics
import ImageIO

func fail(_ message: String, code: Int32) -> Never {
    fputs("read_text: \(message)\n", stderr)
    exit(code)
}

let args = CommandLine.arguments
guard args.count == 5 else {
    fail("usage: read_text.swift <image> <columns> <rows> <tiles>", code: 2)
}
let imagePath = args[1]
guard let columns = Int(args[2]), columns > 0,
    let rows = Int(args[3]), rows > 0,
    let tileCount = Int(args[4]), tileCount > 0
else {
    fail("columns, rows and tiles must be positive integers", code: 2)
}
guard tileCount <= columns * rows else {
    fail(
        "tiles (\(tileCount)) is above columns x rows (\(columns * rows))",
        code: 2
    )
}
let url = URL(fileURLWithPath: imagePath)
guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
    let full = CGImageSourceCreateImageAtIndex(source, 0, nil)
else {
    fail("cannot read image \(imagePath)", code: 1)
}
let width = full.width
let height = full.height
guard width % columns == 0 && height % rows == 0 else {
    fail(
        "uneven grid: \(width)x\(height) does not divide into \(columns)x\(rows)",
        code: 1
    )
}
let tileWidth = width / columns
let tileHeight = height / rows
var revision = 0
var tileObjects: [[String: Any]] = []
for index in 0..<tileCount {
    let column = index % columns
    let row = index / columns
    let rect = CGRect(
        x: column * tileWidth,
        y: row * tileHeight,
        width: tileWidth,
        height: tileHeight
    )
    guard let cropped = full.cropping(to: rect) else {
        fail("cannot crop tile \(index)", code: 1)
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.recognitionLanguages = ["en-US"]
    do {
        try VNImageRequestHandler(cgImage: cropped, options: [:]).perform([request])
    } catch {
        fail("Vision failed on tile \(index): \(error)", code: 1)
    }
    revision = request.revision
    var lineObjects: [[String: Any]] = []
    for observation in request.results ?? [] {
        guard let candidate = observation.topCandidates(1).first else {
            continue
        }
        let box = observation.boundingBox
        lineObjects.append([
            "text": candidate.string,
            "confidence": Double(candidate.confidence),
            "box": [
                Double(box.minX),
                Double(1.0 - box.maxY),
                Double(box.width),
                Double(box.height),
            ],
        ])
    }
    tileObjects.append(["index": index, "lines": lineObjects])
}
let payload: [String: Any] = [
    "engine": "Apple Vision VNRecognizeTextRequest",
    "revision": revision,
    "tiles": tileObjects,
]
do {
    let data = try JSONSerialization.data(
        withJSONObject: payload, options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self))
} catch {
    fail("cannot encode JSON: \(error)", code: 1)
}
