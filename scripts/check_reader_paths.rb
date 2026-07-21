#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "set"

ROOT = File.expand_path("..", __dir__)
INDEX = JSON.parse(File.read(File.join(ROOT, "_data/content-index.json"), encoding: "UTF-8"))
PATH_KEYS = %w[research-teaching notes-writing faith-spirituality projects-creation].freeze
LANGUAGES = %w[zh en].freeze

def rendered_path(path_key, language, archive: false)
  prefix = language == "en" ? "en/" : ""
  suffix = archive ? "/archive" : ""
  File.join(ROOT, "_site/#{prefix}paths/#{path_key}#{suffix}/index.html")
end

def content_ids(html)
  html.scan(/data-content-id="([^"]+)"/).flatten
end

def assert_newest_first!(ids, items_by_id, label)
  dates = ids.map { |content_id| items_by_id.fetch(content_id).fetch("date") }
  raise "items are not newest first in #{label}: #{dates.join(', ')}" unless dates == dates.sort.reverse
end

items = INDEX.fetch("items")
items_by_id = items.to_h { |item| [item.fetch("content_id"), item] }

PATH_KEYS.each do |path_key|
  path_items = items.select { |item| item.fetch("reader_paths").include?(path_key) }
  representative_items = path_items.select { |item| item.fetch("representative_paths", []).include?(path_key) }

  raise "reader path has no representative work: #{path_key}" if representative_items.empty?

  LANGUAGES.each do |language|
    landing_path = rendered_path(path_key, language)
    archive_path = rendered_path(path_key, language, archive: true)
    raise "missing rendered reader path: #{landing_path}" unless File.file?(landing_path)
    raise "missing rendered path archive: #{archive_path}" unless File.file?(archive_path)

    landing_html = File.read(landing_path, encoding: "UTF-8")
    archive_html = File.read(archive_path, encoding: "UTF-8")
    landing_ids = content_ids(landing_html)
    archive_ids = content_ids(archive_html)

    unless landing_ids.to_set == representative_items.map { |item| item.fetch("content_id") }.to_set
      raise "landing representatives do not match metadata for #{path_key}/#{language}"
    end
    unless archive_ids.to_set == path_items.map { |item| item.fetch("content_id") }.to_set
      raise "archive does not match path membership for #{path_key}/#{language}"
    end

    assert_newest_first!(landing_ids, items_by_id, "#{path_key}/#{language}")
    assert_newest_first!(archive_ids, items_by_id, "#{path_key}/#{language}/archive")

    html_language = language == "en" ? "en" : "zh"
    raise "page language is incorrect for #{path_key}/#{language}" unless landing_html.include?(%(<html lang="#{html_language}"))
  end
end

zh_research = File.read(rendered_path("research-teaching", "zh"), encoding: "UTF-8")
raise "Chinese path does not label English originals" unless zh_research.include?("原文：英文")

en_notes = File.read(rendered_path("notes-writing", "en"), encoding: "UTF-8")
raise "English path does not label Chinese originals" unless en_notes.include?("Original: Chinese")

puts "Reader-path static checks: passed for all four bilingual paths."
