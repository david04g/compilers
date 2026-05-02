#include "Json.hpp"

#include <cctype>
#include <sstream>

namespace gpu_sched {
namespace {

class JsonParser {
public:
  explicit JsonParser(const std::string &text) : text_(text) {}

  JsonValue parse() {
    skipWhitespace();
    JsonValue value = parseValue();
    skipWhitespace();
    if (pos_ != text_.size()) {
      fail("unexpected trailing JSON text");
    }
    return value;
  }

private:
  JsonValue parseValue() {
    skipWhitespace();
    if (pos_ >= text_.size()) {
      fail("unexpected end of JSON");
    }
    char c = text_[pos_];
    if (c == '{') {
      return parseObject();
    }
    if (c == '[') {
      return parseArray();
    }
    if (c == '"') {
      return JsonValue(parseString());
    }
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
      return JsonValue(parseNumber());
    }
    if (consume("true")) {
      return JsonValue(true);
    }
    if (consume("false")) {
      return JsonValue(false);
    }
    if (consume("null")) {
      return JsonValue();
    }
    fail("invalid JSON value");
  }

  JsonValue parseObject() {
    expect('{');
    std::map<std::string, JsonValue> object;
    skipWhitespace();
    if (peek('}')) {
      ++pos_;
      return JsonValue::object(std::move(object));
    }
    while (true) {
      skipWhitespace();
      if (!peek('"')) {
        fail("expected object key string");
      }
      std::string key = parseString();
      skipWhitespace();
      expect(':');
      object.emplace(std::move(key), parseValue());
      skipWhitespace();
      if (peek('}')) {
        ++pos_;
        break;
      }
      expect(',');
    }
    return JsonValue::object(std::move(object));
  }

  JsonValue parseArray() {
    expect('[');
    std::vector<JsonValue> values;
    skipWhitespace();
    if (peek(']')) {
      ++pos_;
      return JsonValue::array(std::move(values));
    }
    while (true) {
      values.push_back(parseValue());
      skipWhitespace();
      if (peek(']')) {
        ++pos_;
        break;
      }
      expect(',');
    }
    return JsonValue::array(std::move(values));
  }

  std::string parseString() {
    expect('"');
    std::string result;
    while (pos_ < text_.size()) {
      char c = text_[pos_++];
      if (c == '"') {
        return result;
      }
      if (c == '\\') {
        if (pos_ >= text_.size()) {
          fail("unterminated JSON escape");
        }
        char escaped = text_[pos_++];
        switch (escaped) {
        case '"':
        case '\\':
        case '/':
          result.push_back(escaped);
          break;
        case 'b':
          result.push_back('\b');
          break;
        case 'f':
          result.push_back('\f');
          break;
        case 'n':
          result.push_back('\n');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case 't':
          result.push_back('\t');
          break;
        default:
          fail("unsupported JSON escape");
        }
      } else {
        result.push_back(c);
      }
    }
    fail("unterminated JSON string");
  }

  double parseNumber() {
    size_t start = pos_;
    if (peek('-')) {
      ++pos_;
    }
    while (pos_ < text_.size() &&
           std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
    if (peek('.')) {
      ++pos_;
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    return std::stod(text_.substr(start, pos_ - start));
  }

  bool consume(const char *keyword) {
    std::string word(keyword);
    if (text_.compare(pos_, word.size(), word) == 0) {
      pos_ += word.size();
      return true;
    }
    return false;
  }

  bool peek(char c) const { return pos_ < text_.size() && text_[pos_] == c; }

  void expect(char c) {
    skipWhitespace();
    if (!peek(c)) {
      std::string message = "expected '";
      message.push_back(c);
      message.push_back('\'');
      fail(message);
    }
    ++pos_;
  }

  void skipWhitespace() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  [[noreturn]] void fail(const std::string &message) const {
    std::ostringstream out;
    out << message << " at byte " << pos_;
    throw JsonError(out.str());
  }

  const std::string &text_;
  size_t pos_ = 0;
};

} // namespace

JsonValue::JsonValue()
    : type_(Type::Null), bool_value_(false), number_value_(0.0) {}

JsonValue::JsonValue(bool value)
    : type_(Type::Bool), bool_value_(value), number_value_(0.0) {}

JsonValue::JsonValue(double value)
    : type_(Type::Number), bool_value_(false), number_value_(value) {}

JsonValue::JsonValue(std::string value)
    : type_(Type::String), bool_value_(false), number_value_(0.0),
      string_value_(std::move(value)) {}

JsonValue JsonValue::array(std::vector<JsonValue> values) {
  JsonValue value;
  value.type_ = Type::Array;
  value.array_value_ = std::move(values);
  return value;
}

JsonValue JsonValue::object(std::map<std::string, JsonValue> values) {
  JsonValue value;
  value.type_ = Type::Object;
  value.object_value_ = std::move(values);
  return value;
}

bool JsonValue::asBool() const {
  if (type_ != Type::Bool) {
    throw JsonError("expected JSON bool");
  }
  return bool_value_;
}

double JsonValue::asNumber() const {
  if (type_ != Type::Number) {
    throw JsonError("expected JSON number");
  }
  return number_value_;
}

int JsonValue::asInt() const { return static_cast<int>(asNumber()); }

const std::string &JsonValue::asString() const {
  if (type_ != Type::String) {
    throw JsonError("expected JSON string");
  }
  return string_value_;
}

const std::vector<JsonValue> &JsonValue::asArray() const {
  if (type_ != Type::Array) {
    throw JsonError("expected JSON array");
  }
  return array_value_;
}

const std::map<std::string, JsonValue> &JsonValue::asObject() const {
  if (type_ != Type::Object) {
    throw JsonError("expected JSON object");
  }
  return object_value_;
}

bool JsonValue::contains(const std::string &key) const {
  return type_ == Type::Object && object_value_.find(key) != object_value_.end();
}

const JsonValue &JsonValue::at(const std::string &key) const {
  if (type_ != Type::Object) {
    throw JsonError("expected JSON object");
  }
  auto it = object_value_.find(key);
  if (it == object_value_.end()) {
    throw JsonError("missing JSON key: " + key);
  }
  return it->second;
}

JsonValue parseJson(const std::string &text) { return JsonParser(text).parse(); }

std::string escapeJson(const std::string &text) {
  std::string escaped;
  for (char c : text) {
    switch (c) {
    case '"':
      escaped += "\\\"";
      break;
    case '\\':
      escaped += "\\\\";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped.push_back(c);
      break;
    }
  }
  return escaped;
}

} // namespace gpu_sched
