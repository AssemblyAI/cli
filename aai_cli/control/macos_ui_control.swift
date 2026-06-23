import AppKit
import ApplicationServices
import CoreGraphics
import Foundation

// A tiny JSON-lines UI-control helper: read one request object per stdin line,
// perform the action with native macOS APIs (CGEvent for synthetic input, the
// Accessibility API for the element tree, NSWorkspace for app launch/focus), and
// write one JSON result line per request. Python (aai_cli/control/helper.py) owns
// the lifecycle and speaks this protocol; see that module for the request shape.

// Maps element ids handed out by get_ui_tree back to their AXUIElement, so a
// later click can target one by id rather than by guessed coordinates.
var elementRegistry: [String: AXUIElement] = [:]

// US-keyboard virtual key codes for the keys key_combo can press.
let keyCodes: [String: CGKeyCode] = [
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19,
    "3": 20, "4": 21, "6": 22, "5": 23, "9": 25, "7": 26, "8": 28, "0": 29, "o": 31,
    "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45, "m": 46,
    "return": 36, "enter": 36, "tab": 48, "space": 49, "delete": 51, "backspace": 51,
    "escape": 53, "esc": 53, "left": 123, "right": 124, "down": 125, "up": 126,
    "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
]

// Modifier names key_combo accepts, mapped to CGEvent flags.
let modifierFlags: [String: CGEventFlags] = [
    "cmd": .maskCommand, "command": .maskCommand, "meta": .maskCommand,
    "shift": .maskShift,
    "ctrl": .maskControl, "control": .maskControl,
    "alt": .maskAlternate, "option": .maskAlternate, "opt": .maskAlternate,
    "fn": .maskSecondaryFn,
]

// One request line: the action name plus every argument any action may carry
// (all optional; each handler reads the ones it needs). Decoding ignores extra
// keys, so the protocol can grow additively.
struct Request: Decodable {
    let action: String
    let text: String?
    let keys: [String]?
    let name: String?
    let element: String?
    let x: Int?
    let y: Int?
}

// One labeled, clickable accessibility element reported by get_ui_tree.
struct Element: Encodable {
    let id: String
    let role: String
    let title: String
    let x: Int?
    let y: Int?
}

// One result line. nil fields are omitted by JSONEncoder, so a plain success is
// just {"ok": true} and an element list / screenshot path appears only when set.
struct Response: Encodable {
    var ok: Bool
    var error: String?
    var elements: [Element]?
    var path: String?
}

func succeeded() -> Response {
    return Response(ok: true, error: nil, elements: nil, path: nil)
}

func failure(_ message: String) -> Response {
    return Response(ok: false, error: message, elements: nil, path: nil)
}

func emit(_ response: Response) {
    guard
        let data = try? JSONEncoder().encode(response),
        let text = String(data: data, encoding: .utf8)
    else {
        FileHandle.standardError.write(Data("failed to encode helper response\n".utf8))
        return
    }
    print(text)
    fflush(stdout)
}

func typeText(_ text: String) -> Response {
    let source = CGEventSource(stateID: .combinedSessionState)
    guard
        let down = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: true),
        let up = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: false)
    else {
        return failure("could not create keyboard event")
    }
    let utf16 = Array(text.utf16)
    utf16.withUnsafeBufferPointer { buffer in
        if let base = buffer.baseAddress {
            down.keyboardSetUnicodeString(stringLength: buffer.count, unicodeString: base)
            up.keyboardSetUnicodeString(stringLength: buffer.count, unicodeString: base)
        }
    }
    down.post(tap: .cghidEventTap)
    up.post(tap: .cghidEventTap)
    return succeeded()
}

func keyCombo(_ keys: [String]) -> Response {
    var flags: CGEventFlags = []
    var mainKey: CGKeyCode?
    for key in keys {
        let lower = key.lowercased()
        if let flag = modifierFlags[lower] {
            flags.insert(flag)
        } else if let code = keyCodes[lower] {
            mainKey = code
        } else {
            return failure("unknown key: \(key)")
        }
    }
    guard let code = mainKey else {
        return failure("key_combo needs one non-modifier key")
    }
    let source = CGEventSource(stateID: .combinedSessionState)
    guard
        let down = CGEvent(keyboardEventSource: source, virtualKey: code, keyDown: true),
        let up = CGEvent(keyboardEventSource: source, virtualKey: code, keyDown: false)
    else {
        return failure("could not create keyboard event")
    }
    down.flags = flags
    up.flags = flags
    down.post(tap: .cghidEventTap)
    up.post(tap: .cghidEventTap)
    return succeeded()
}

func frontmostApp() -> AXUIElement? {
    guard let app = NSWorkspace.shared.frontmostApplication else {
        return nil
    }
    return AXUIElementCreateApplication(app.processIdentifier)
}

func copyAttribute(_ element: AXUIElement, _ attribute: String) -> CFTypeRef? {
    var value: CFTypeRef?
    let status = AXUIElementCopyAttributeValue(element, attribute as CFString, &value)
    guard status == .success else {
        return nil
    }
    return value
}

func childElements(_ element: AXUIElement) -> [AXUIElement] {
    guard let raw = copyAttribute(element, kAXChildrenAttribute as String) else {
        return []
    }
    return (raw as? [AXUIElement]) ?? []
}

func stringAttribute(_ element: AXUIElement, _ attribute: String) -> String? {
    guard let value = copyAttribute(element, attribute) else {
        return nil
    }
    return value as? String
}

func elementFrame(_ element: AXUIElement) -> CGRect? {
    guard
        let positionValue = copyAttribute(element, kAXPositionAttribute as String),
        let sizeValue = copyAttribute(element, kAXSizeAttribute as String),
        CFGetTypeID(positionValue) == AXValueGetTypeID(),
        CFGetTypeID(sizeValue) == AXValueGetTypeID()
    else {
        return nil
    }
    let position = positionValue as! AXValue
    let size = sizeValue as! AXValue
    var point = CGPoint.zero
    var dimensions = CGSize.zero
    guard
        AXValueGetValue(position, .cgPoint, &point),
        AXValueGetValue(size, .cgSize, &dimensions)
    else {
        return nil
    }
    return CGRect(origin: point, size: dimensions)
}

func buildTree() -> Response {
    guard AXIsProcessTrusted() else {
        return failure(
            "Accessibility permission is required. Grant it in System Settings > "
                + "Privacy & Security > Accessibility."
        )
    }
    guard let app = frontmostApp() else {
        return failure("no frontmost application")
    }
    elementRegistry.removeAll()
    var collected: [Element] = []
    var queue: [AXUIElement] = [app]
    var index = 0
    let maxElements = 200
    while !queue.isEmpty && collected.count < maxElements {
        let element = queue.removeFirst()
        queue.append(contentsOf: childElements(element))
        let role = stringAttribute(element, kAXRoleAttribute as String) ?? ""
        let label =
            stringAttribute(element, kAXTitleAttribute as String)
            ?? stringAttribute(element, kAXDescriptionAttribute as String)
            ?? stringAttribute(element, kAXValueAttribute as String)
        guard !role.isEmpty, let title = label, !title.isEmpty else {
            continue
        }
        let identifier = "e\(index)"
        index += 1
        elementRegistry[identifier] = element
        let rect = elementFrame(element)
        collected.append(
            Element(
                id: identifier,
                role: role,
                title: title,
                x: rect.map { Int($0.midX) },
                y: rect.map { Int($0.midY) }
            )
        )
    }
    return Response(ok: true, error: nil, elements: collected, path: nil)
}

func clickAt(x: CGFloat, y: CGFloat) -> Response {
    let point = CGPoint(x: x, y: y)
    let source = CGEventSource(stateID: .combinedSessionState)
    guard
        let down = CGEvent(
            mouseEventSource: source, mouseType: .leftMouseDown,
            mouseCursorPosition: point, mouseButton: .left),
        let up = CGEvent(
            mouseEventSource: source, mouseType: .leftMouseUp,
            mouseCursorPosition: point, mouseButton: .left)
    else {
        return failure("could not create mouse event")
    }
    down.post(tap: .cghidEventTap)
    up.post(tap: .cghidEventTap)
    return succeeded()
}

func click(_ request: Request) -> Response {
    if let identifier = request.element {
        guard let element = elementRegistry[identifier] else {
            return failure("unknown element id \(identifier); call get_ui_tree first")
        }
        if AXUIElementPerformAction(element, kAXPressAction as CFString) == .success {
            return succeeded()
        }
        guard let rect = elementFrame(element) else {
            return failure("could not locate element \(identifier)")
        }
        return clickAt(x: rect.midX, y: rect.midY)
    }
    if let x = request.x, let y = request.y {
        return clickAt(x: CGFloat(x), y: CGFloat(y))
    }
    return failure("click needs an element id or x/y coordinates")
}

func launchApp(_ name: String) -> Response {
    if NSWorkspace.shared.launchApplication(name) {
        return succeeded()
    }
    return failure("could not launch application: \(name)")
}

func focusApp(_ name: String) -> Response {
    let lower = name.lowercased()
    for app in NSWorkspace.shared.runningApplications where app.localizedName?.lowercased() == lower
    {
        app.activate(options: [.activateAllWindows])
        return succeeded()
    }
    return failure("application not running: \(name)")
}

func screenshot() -> Response {
    guard let image = CGDisplayCreateImage(CGMainDisplayID()) else {
        return failure("could not capture the screen; grant Screen Recording permission")
    }
    let bitmap = NSBitmapImageRep(cgImage: image)
    guard let data = bitmap.representation(using: .png, properties: [:]) else {
        return failure("could not encode the screenshot")
    }
    let path = NSTemporaryDirectory() + "aai-control-screenshot.png"
    do {
        try data.write(to: URL(fileURLWithPath: path))
    } catch {
        return failure("could not save the screenshot: \(error)")
    }
    return Response(ok: true, error: nil, elements: nil, path: path)
}

func handle(_ request: Request) -> Response {
    switch request.action {
    case "type_text":
        guard let text = request.text else {
            return failure("type_text needs 'text'")
        }
        return typeText(text)
    case "key_combo":
        guard let keys = request.keys else {
            return failure("key_combo needs 'keys'")
        }
        return keyCombo(keys)
    case "click":
        return click(request)
    case "launch_app":
        guard let name = request.name else {
            return failure("launch_app needs 'name'")
        }
        return launchApp(name)
    case "focus_app":
        guard let name = request.name else {
            return failure("focus_app needs 'name'")
        }
        return focusApp(name)
    case "get_ui_tree":
        return buildTree()
    case "screenshot":
        return screenshot()
    default:
        return failure("unknown action: \(request.action)")
    }
}

@main
struct Main {
    static func main() {
        let decoder = JSONDecoder()
        while let line = readLine(strippingNewline: true) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                continue
            }
            guard
                let data = trimmed.data(using: .utf8),
                let request = try? decoder.decode(Request.self, from: data)
            else {
                emit(failure("invalid JSON request"))
                continue
            }
            emit(handle(request))
        }
    }
}
