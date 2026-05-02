#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_sched {

class JsonError : public std::runtime_error {
public:
  explicit JsonError(const std::string &message) : std::runtime_error(message) {}
};

class JsonValue {
public:
  enum class Type {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object
  };

  JsonValue();
  explicit JsonValue(bool value);
  explicit JsonValue(double value);
  explicit JsonValue(std::string value);

  static JsonValue array(std::vector<JsonValue> values);
  static JsonValue object(std::map<std::string, JsonValue> values);

  Type type() const { return type_; }
  bool isNull() const { return type_ == Type::Null; }
  bool isObject() const { return type_ == Type::Object; }
  bool isArray() const { return type_ == Type::Array; }

  bool asBool() const;
  double asNumber() const;
  int asInt() const;
  const std::string &asString() const;
  const std::vector<JsonValue> &asArray() const;
  const std::map<std::string, JsonValue> &asObject() const;

  bool contains(const std::string &key) const;
  const JsonValue &at(const std::string &key) const;
  const JsonValue &operator[](const std::string &key) const { return at(key); }

private:
  Type type_;
  bool bool_value_;
  double number_value_;
  std::string string_value_;
  std::vector<JsonValue> array_value_;
  std::map<std::string, JsonValue> object_value_;
};

JsonValue parseJson(const std::string &text);
std::string escapeJson(const std::string &text);

} // namespace gpu_sched
