#!/usr/bin/env ruby
# frozen_string_literal: true

require "cgi"
require "json"
require "pathname"
require "uri"

ROOT = Pathname(File.expand_path("..", __dir__))
INDEX = JSON.parse(ROOT.join("_data/content-index.json").read(encoding: "UTF-8"))
SITE_URL = "https://XavierOwen.github.io"

def rendered_file(url)
  relative = url.delete_prefix("/")
  candidates = if relative.empty? || relative.end_with?("/")
                 [ROOT.join("_site", relative, "index.html")]
               else
                 [ROOT.join("_site", relative), ROOT.join("_site", "#{relative}.html"), ROOT.join("_site", relative, "index.html")]
               end
  candidates.find(&:file?) || raise("missing rendered article for #{url}")
end

def assert_includes!(html, value, label)
  raise "#{label}: missing #{value}" unless html.include?(value)
end

def decoded_href(html, relation:, hreflang: nil)
  pattern = if hreflang
              /<link rel="#{Regexp.escape(relation)}" hreflang="#{Regexp.escape(hreflang)}" href="([^"]+)">/
            else
              /<link rel="#{Regexp.escape(relation)}" href="([^"]+)">/
            end
  href = html[pattern, 1]
  href && URI::DEFAULT_PARSER.unescape(CGI.unescape_html(href))
end

INDEX.fetch("items").each do |item|
  versions = item.fetch("versions")
  versions.each do |language, version|
    url = version.fetch("url")
    path = rendered_file(url)
    html = path.read(encoding: "UTF-8")

    assert_includes!(html, %(<html lang="#{language}"), path)
    assert_includes!(html, %(data-language-context="article"), path)
    assert_includes!(html, %(data-language-current="#{language}"), path)
    canonical_url = decoded_href(html, relation: "canonical")
    raise "#{path}: canonical is not self-referential" unless canonical_url == "#{SITE_URL}#{url}"
    assert_includes!(html, ">中</a>", path)
    assert_includes!(html, ">EN</a>", path)

    if versions.keys.sort == %w[en zh]
      %w[zh en].each do |alternate_language|
        alternate_url = versions.fetch(alternate_language).fetch("url")
        rendered_alternate_url = decoded_href(html, relation: "alternate", hreflang: alternate_language)
        unless rendered_alternate_url == "#{SITE_URL}#{alternate_url}"
          raise "#{path}: incorrect #{alternate_language} alternate-language URL"
        end
      end
    elsif html.match?(/rel="alternate" hreflang="(?:zh|en)"/)
      raise "#{path}: unpaired article emitted alternate-language metadata"
    end
  end
end

chinese_only = INDEX.fetch("items").find { |item| item.fetch("versions").keys == ["zh"] }
english_only = INDEX.fetch("items").find { |item| item.fetch("versions").keys == ["en"] }
raise "fixture corpus needs a Chinese-only article" unless chinese_only
raise "fixture corpus needs an English-only article" unless english_only

chinese_html = rendered_file(chinese_only.dig("versions", "zh", "url")).read(encoding: "UTF-8")
english_html = rendered_file(english_only.dig("versions", "en", "url")).read(encoding: "UTF-8")
assert_includes!(chinese_html, %(data-en-available="false"), "Chinese fallback contract")
assert_includes!(chinese_html, "The Chinese original is shown.", "English UI fallback copy")
assert_includes!(english_html, %(data-zh-available="false"), "English fallback contract")
assert_includes!(english_html, "现显示英文原文", "Chinese UI fallback copy")

puts "Bilingual article static checks: passed for #{INDEX.fetch('items').length} conceptual items."
