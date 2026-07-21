#!/usr/bin/env ruby
# frozen_string_literal: true

require "optparse"
require "date"
require "pathname"
require "yaml"

class PublishedContentAudit
  COLLECTIONS = %w[notes spirits projects publications teaching posts].freeze
  LANGUAGES = %w[zh en].freeze
  PATH_KEYS = %w[
    research-teaching
    notes-writing
    faith-spirituality
    projects-creation
  ].freeze
  REQUIRED_FIELDS = %w[
    content_id
    language
    original_language
    reader_paths
    title_zh
    title_en
    summary_zh
    summary_en
  ].freeze

  Document = Struct.new(:path, :collection, :data, :content, keyword_init: true)

  def initialize(root:, report_only: false)
    @root = Pathname(root)
    @report_only = report_only
  end

  def run
    print_report(errors)
    return 0 if errors.empty? || @report_only

    1
  end

  def errors
    @errors ||= documents.flat_map { |document| validate_document(document) } + validate_translation_groups
  end

  def documents
    @documents ||= COLLECTIONS.flat_map do |collection|
      directory = @root.join("_#{collection}")
      next [] unless directory.directory?

      directory.glob("**/*.md").sort.filter_map do |path|
        data, content = parse_document(path)
        next if data["published"] == false

        Document.new(path: path.relative_path_from(@root).to_s, collection: collection, data: data, content: content)
      end
    end
  end

  def validate_document(document)
    data = document.data
    errors = REQUIRED_FIELDS.filter_map do |field|
      "#{document.path}: missing #{field}" if blank?(data[field])
    end
    return errors unless errors.empty?

    unless LANGUAGES.include?(data["language"])
      errors << "#{document.path}: language must be one of #{LANGUAGES.join(", ")}"
    end
    unless LANGUAGES.include?(data["original_language"])
      errors << "#{document.path}: original_language must be one of #{LANGUAGES.join(", ")}"
    end

    reader_paths = array_value(data["reader_paths"])
    invalid_paths = reader_paths - PATH_KEYS
    errors << "#{document.path}: reader_paths contains unsupported keys: #{invalid_paths.join(", ")}" unless invalid_paths.empty?
    errors << "#{document.path}: reader_paths must contain at least one path" if reader_paths.empty?

    representative_paths = array_value(data["representative_paths"])
    invalid_representatives = representative_paths - reader_paths
    unless invalid_representatives.empty?
      errors << "#{document.path}: representative_paths must be a subset of reader_paths: #{invalid_representatives.join(", ")}"
    end

    %w[title_zh title_en summary_zh summary_en].each do |field|
      errors << "#{document.path}: #{field} must be a non-empty string" unless data[field].is_a?(String) && !data[field].strip.empty?
    end
    errors
  end

  def validate_translation_groups
    valid_documents = documents.reject { |document| validate_document(document).any? }
    valid_documents.group_by { |document| document.data["content_id"] }.flat_map do |content_id, group|
      languages = group.map { |document| document.data["language"] }
      errors = []
      duplicate_languages = languages.tally.select { |_language, count| count > 1 }.keys
      errors << "content_id #{content_id}: duplicate language versions: #{duplicate_languages.join(", ")}" unless duplicate_languages.empty?

      originals = group.select { |document| document.data["language"] == document.data["original_language"] }
      errors << "content_id #{content_id}: must have exactly one original-language document" unless originals.length == 1

      %w[original_language reader_paths representative_paths title_zh title_en summary_zh summary_en].each do |field|
        values = group.map { |document| normalized_group_value(field, document.data[field]) }.uniq
        errors << "content_id #{content_id}: #{field} must match across language versions" if values.length > 1
      end

      group.reject { |document| document.data["language"] == document.data["original_language"] }.each do |translation|
        unless translation.data["translation_reviewed"] == true
          errors << "#{translation.path}: translation_reviewed must be true for a translation"
        end
      end
      errors
    end
  end

  private

  def normalized_group_value(field, value)
    %w[reader_paths representative_paths].include?(field) ? array_value(value).sort : value
  end

  def parse_document(path)
    source = path.read(encoding: "UTF-8")
    match = source.match(/\A---\s*\n(.*?)\n---\s*(?:\n|\z)/m)
    return [{}, source] unless match

    data = YAML.safe_load(match[1], permitted_classes: [Date, Time], aliases: false) || {}
    [data, source[match.end(0)..].to_s]
  rescue Psych::Exception => error
    [{ "_front_matter_error" => error.message }, source]
  end

  def blank?(value)
    value.nil? || (value.respond_to?(:empty?) && value.empty?)
  end

  def array_value(value)
    value.is_a?(Array) ? value.map(&:to_s) : []
  end

  def print_report(errors)
    puts "Audited #{documents.length} published items across #{COLLECTIONS.join(", ")}."
    if errors.empty?
      puts "Published-content contract: valid."
      return
    end

    puts "Published-content contract: #{errors.length} violation(s)."
    errors.each { |error| puts "- #{error}" }
    puts "Report-only mode: violations do not change the exit status." if @report_only
  end
end

def self_test
  valid = PublishedContentAudit::Document.new(
    path: "_notes/valid.md",
    collection: "notes",
    data: {
      "content_id" => "valid-note", "language" => "zh", "original_language" => "zh",
      "reader_paths" => ["notes-writing"], "representative_paths" => ["notes-writing"],
      "title_zh" => "有效", "title_en" => "Valid", "summary_zh" => "简介", "summary_en" => "Summary"
    }
  )
  invalid = PublishedContentAudit::Document.new(path: "_notes/invalid.md", collection: "notes", data: {})
  translation = PublishedContentAudit::Document.new(
    path: "_notes/valid-en.md",
    collection: "notes",
    data: valid.data.merge("language" => "en", "translation_reviewed" => true)
  )
  unreviewed_translation = PublishedContentAudit::Document.new(
    path: "_notes/unreviewed-en.md",
    collection: "notes",
    data: valid.data.merge("language" => "en")
  )
  audit = PublishedContentAudit.new(root: Dir.pwd)
  raise "valid fixture failed" unless audit.validate_document(valid).empty?
  raise "invalid fixture passed" if audit.validate_document(invalid).empty?
  audit.instance_variable_set(:@documents, [valid, translation])
  raise "valid translation pair failed" unless audit.validate_translation_groups.empty?
  audit.instance_variable_set(:@documents, [valid, unreviewed_translation])
  raise "unreviewed translation passed" if audit.validate_translation_groups.empty?
  puts "Published-content audit self-test: passed."
end

if __FILE__ == $PROGRAM_NAME
  if ARGV.delete("--self-test")
    self_test
    exit 0
  end

  options = { report_only: false }
  OptionParser.new do |parser|
    parser.banner = "Usage: ruby scripts/audit_published_content.rb [--report]"
    parser.on("--report", "Print violations without failing") { options[:report_only] = true }
  end.parse!

  exit PublishedContentAudit.new(root: Dir.pwd, report_only: options[:report_only]).run
end
