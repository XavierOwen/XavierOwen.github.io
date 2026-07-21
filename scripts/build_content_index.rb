#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "pathname"
require_relative "audit_published_content"

class ContentIndexBuilder
  SCHEMA_VERSION = 1

  def initialize(root:, documents: nil)
    @root = Pathname(root)
    @audit = PublishedContentAudit.new(root: @root)
    @documents = documents
  end

  def build
    documents = @documents || validated_documents
    items = documents.group_by { |document| document.data.fetch("content_id") }.sort.map do |content_id, group|
      build_item(content_id, group)
    end
    attach_backlinks(items, documents)

    { "schema_version" => SCHEMA_VERSION, "items" => items }
  end

  def generate(output: @root.join("_data/content-index.json"))
    json = JSON.pretty_generate(build) + "\n"
    output = Pathname(output)
    output.dirname.mkpath
    output.write(json, encoding: "UTF-8") unless output.file? && output.read(encoding: "UTF-8") == json
    puts "Generated #{output.relative_path_from(@root)} with #{build.fetch("items").length} conceptual items."
  end

  private

  def validated_documents
    return @audit.documents if @audit.errors.empty?

    raise "Published-content contract is invalid; run npm run audit:content for details."
  end

  def build_item(content_id, group)
    source = group.find { |document| document.data["language"] == document.data["original_language"] }
    data = source.data
    {
      "content_id" => content_id,
      "date" => data["date"]&.to_s,
      "original_language" => data.fetch("original_language"),
      "reader_paths" => Array(data["reader_paths"]).map(&:to_s).sort,
      "representative_paths" => Array(data["representative_paths"]).map(&:to_s).sort,
      "aliases" => wiki_aliases(group),
      "titles" => { "zh" => data.fetch("title_zh"), "en" => data.fetch("title_en") },
      "summaries" => { "zh" => data.fetch("summary_zh"), "en" => data.fetch("summary_en") },
      "versions" => group.sort_by { |document| document.data.fetch("language") }.to_h do |document|
        language = document.data.fetch("language")
        [language, build_version(document, language)]
      end
    }
  end

  def build_version(document, language)
    data = document.data
    {
      "language" => language,
      "title" => data["title"] || data.fetch("title_#{language}"),
      "summary" => data.fetch("summary_#{language}"),
      "url" => public_url(document),
      "date" => data["date"]&.to_s,
      "collection" => document.collection,
      "source_path" => document.path
    }.compact
  end

  def public_url(document)
    permalink = document.data["permalink"]
    return permalink.to_s if permalink && !permalink.to_s.empty?

    path = document.path.delete_prefix("_#{document.collection}/").sub(/\.(?:md|html)\z/, "")
    "/#{document.collection}/#{path}/"
  end

  def wiki_aliases(group)
    group.flat_map do |document|
      document.data.values_at("title", "title_zh", "title_en")
    end.filter_map do |value|
      value.to_s.strip unless value.to_s.strip.empty?
    end.uniq.sort
  end

  def attach_backlinks(items, documents)
    items_by_id = items.to_h { |item| [item.fetch("content_id"), item] }
    aliases = {}

    items.each do |item|
      item.fetch("aliases").each do |title|
        key = normalize_alias(title)
        existing = aliases[key]
        if existing && existing != item.fetch("content_id")
          raise "ambiguous wiki-link alias #{title.inspect}: #{existing} and #{item.fetch('content_id')}"
        end
        aliases[key] = item.fetch("content_id")
      end
    end

    backlinks = Hash.new { |hash, key| hash[key] = [] }
    documents.each do |document|
      source_id = document.data.fetch("content_id")
      wiki_link_titles(document.content).each do |title|
        target_id = aliases[normalize_alias(title)]
        next unless target_id && target_id != source_id

        backlinks[target_id] << source_id
      end
    end

    items.each do |item|
      source_ids = backlinks[item.fetch("content_id")].uniq
      item["backlinks"] = source_ids.sort_by do |source_id|
        source = items_by_id.fetch(source_id)
        [source.fetch("date").to_s, source_id]
      end.reverse
    end
  end

  def wiki_link_titles(content)
    content.to_s.scan(/\[\[([^\]\n]+)\]\]/).flatten.filter_map do |token|
      title = token.strip
      title unless title.empty? || title.include?("::")
    end
  end

  def normalize_alias(title)
    title.to_s.strip.downcase
  end
end

def self_test
  common = {
    "content_id" => "example", "original_language" => "zh",
    "reader_paths" => ["notes-writing"], "representative_paths" => [],
    "title_zh" => "示例", "title_en" => "Example",
    "summary_zh" => "示例简介", "summary_en" => "Example summary",
    "title" => "示例原题"
  }
  documents = [
    PublishedContentAudit::Document.new(
      path: "_notes/example.md", collection: "notes",
      data: common.merge("language" => "zh", "date" => Date.new(2026, 1, 2)),
      content: "See [[Target title]]."
    ),
    PublishedContentAudit::Document.new(
      path: "_notes/example-en.md", collection: "notes",
      data: common.merge("language" => "en", "translation_reviewed" => true, "permalink" => "/en/example/", "title" => "Example authored title"),
      content: "See [[Target title]] again."
    ),
    PublishedContentAudit::Document.new(
      path: "_notes/target.md", collection: "notes",
      data: {
        "content_id" => "target", "language" => "en", "original_language" => "en",
        "reader_paths" => ["notes-writing"], "representative_paths" => [],
        "title" => "Target title", "title_zh" => "目标", "title_en" => "Target",
        "summary_zh" => "目标简介", "summary_en" => "Target summary",
        "date" => Date.new(2025, 1, 1)
      },
      content: ""
    )
  ]
  index = ContentIndexBuilder.new(root: Dir.pwd, documents: documents).build
  raise "unexpected item count" unless index.fetch("items").length == 2
  item = index.fetch("items").find { |entry| entry.fetch("content_id") == "example" }
  raise "languages were not grouped" unless item.fetch("versions").keys == %w[en zh]
  raise "explicit permalink was lost" unless item.dig("versions", "en", "url") == "/en/example/"
  raise "collection URL was not derived" unless item.dig("versions", "zh", "url") == "/notes/example/"
  raise "authored titles were not retained as aliases" unless item.fetch("aliases").include?("Example authored title")
  target = index.fetch("items").find { |entry| entry.fetch("content_id") == "target" }
  raise "translation-pair backlink was not deduplicated" unless target.fetch("backlinks") == ["example"]
  puts "Content-index self-test: passed."
end

if __FILE__ == $PROGRAM_NAME
  if ARGV.delete("--self-test")
    self_test
    exit 0
  end

  ContentIndexBuilder.new(root: Dir.pwd).generate
end
