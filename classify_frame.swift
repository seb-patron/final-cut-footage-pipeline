// Local Apple Vision classifier. Suggestions only, never approval or identity/location evidence.
import Foundation
import Vision
if CommandLine.arguments.count != 2 {
    fputs("Usage: swift classify_frame.swift /absolute/path/frame.jpg\n", stderr)
    exit(2)
}
let url = URL(fileURLWithPath: CommandLine.arguments[1])
let request = VNClassifyImageRequest()
request.usesCPUOnly = true
do {
    try VNImageRequestHandler(url: url, options: [:]).perform([request])
    let rows = (request.results ?? []).prefix(10).map {
        ["label": $0.identifier, "confidence": $0.confidence] as [String: Any]
    }
    let result: [String: Any] = ["engine": "Apple Vision VNClassifyImageRequest", "revision": request.revision,
        "reviewed": false, "approved": false, "results": rows]
    let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
    print(String(decoding: data, as: UTF8.self))
} catch {
    fputs("Local image classification failed: \(error)\n", stderr)
    exit(1)
}
