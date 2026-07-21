#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "pathname"

ROOT = Pathname(File.expand_path("..", __dir__))
INDEX = JSON.parse(ROOT.join("_data/content-index.json").read(encoding: "UTF-8"))

aliases = {}
INDEX.fetch("items").each do |item|
  item.fetch("aliases").each do |title|
    key = title.strip.downcase
    existing = aliases[key]
    raise "ambiguous generated wiki alias: #{title}" if existing && existing != item.fetch("content_id")

    aliases[key] = item.fetch("content_id")
  end

  backlinks = item.fetch("backlinks")
  raise "duplicate backlinks for #{item.fetch('content_id')}" unless backlinks == backlinks.uniq
end

sample_url = INDEX.fetch("items").first.dig("versions").values.first.fetch("url")
relative = sample_url.delete_prefix("/")
candidates = relative.end_with?("/") ? [ROOT.join("_site", relative, "index.html")] : [ROOT.join("_site", relative), ROOT.join("_site", "#{relative}.html")]
sample_page = candidates.find(&:file?) || raise("missing sample article for embedded wiki index")
html = sample_page.read(encoding: "UTF-8")
embedded_json = html[/<script type="application\/json" id="wiki-link-index"[^>]*>(.*?)<\/script>/m, 1]
raise "article does not embed the generated wiki index" unless embedded_json

embedded = JSON.parse(embedded_json.gsub("<\\/", "</"))
raise "embedded wiki index drifted from generated data" unless embedded.fetch("items").length == INDEX.fetch("items").length

puts "Wiki graph static checks: passed for #{aliases.length} aliases."
